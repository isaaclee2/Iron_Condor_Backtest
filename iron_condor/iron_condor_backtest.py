import os
import zstandard as zstd
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
import webbrowser
import re
import matplotlib.pyplot as plt

DTE = 3
ENTRY_TIME = 103000
CALL_DELTA_TARGET = 0.3
PUT_DELTA_TARGET = 0.3
SPREAD = 150
US_business_days= CustomBusinessDay(calendar=USFederalHolidayCalendar())

INITIAL_CAPITAL = 100000.0
CONTRACTS_PER_TRADE = 1

def add_trading_days(date, days_to_add=DTE):
    date = pd.to_datetime(date)
    target_date = date + (days_to_add * US_business_days)
    return target_date.strftime('%Y%m%d')

def find_call_options(df, delta_target=CALL_DELTA_TARGET, spread=SPREAD):
    calls_df = df[df['option_type'] == 'C'].copy()
    calls_below_target = calls_df[calls_df['delta'] <= delta_target]
    if calls_below_target.empty:
        return None, None
    short_call_option = calls_below_target.loc[calls_below_target['delta'].idxmax()]

    target_long_strike = short_call_option['strike'] + spread
    calls_df['strike_diff'] = abs(calls_df['strike'] - target_long_strike)
    long_call_option = calls_df.loc[calls_df['strike_diff'].idxmin()]

    return long_call_option, short_call_option # long is bought, short is sold

def find_put_options(df, delta_target=PUT_DELTA_TARGET, spread=SPREAD):
    puts_df = df[df['option_type'] == 'P'].copy()
    puts_df['delta'] = abs(puts_df['delta'])
    puts_below_target = puts_df[puts_df['delta'] <= delta_target]
    if puts_below_target.empty:
        return None, None
    short_put_option = puts_below_target.loc[puts_below_target['delta'].idxmax()]

    target_long_strike = short_put_option['strike'] - spread
    puts_df['strike_diff'] = abs(puts_df['strike'] - target_long_strike)
    long_put_option = puts_df.loc[puts_df['strike_diff'].idxmin()]

    return long_put_option, short_put_option

class IronCondor:
    def __init__(self, entry_date, expiration_date, long_call, short_call, long_put, short_put):
        self.entry_date = entry_date
        self.expiration_date = expiration_date
        self.long_call = long_call
        self.short_call = short_call
        self.long_put = long_put
        self.short_put = short_put
        self.call_spread = abs(long_call['strike'] - short_call['strike'])
        self.put_spread = abs(long_put['strike'] - short_put['strike'])
        self.net_premium = self.calculate_net_premium()
    
    def calculate_net_premium(self):
        short_call_premium = self.short_call['bid']
        short_put_premium = self.short_put['bid']
        long_call_premium = self.long_call['ask']
        long_put_premium = self.long_put['ask']
        return (short_call_premium + short_put_premium) - (long_call_premium + long_put_premium)     
    
    def display(self):
        print("="*50)
        print(f"Entry Date: {self.entry_date}")
        print(f"Short Call: Strike: {self.short_call['strike']}, Delta: {self.short_call['delta']}")
        print(f"Long Call:  Strike: {self.long_call['strike']}, Delta: {self.long_call['delta']}")
        print(f"Short Put:  Strike: {self.short_put['strike']}, Delta: {self.short_put['delta']}")
        print(f"Long Put:   Strike: {self.long_put['strike']}, Delta: {self.long_put['delta']}")
        print(f"Net Premium: {self.net_premium}")

    def to_dict(self):
        return {
            'entry_date': self.entry_date,
            'expiration_date': self.expiration_date,
            'short_call_strike': self.short_call['strike'],
            'long_call_strike': self.long_call['strike'],
            'short_call_delta': self.short_call['delta'],
            'long_call_delta': self.long_call['delta'],
            'short_put_strike': self.short_put['strike'],
            'long_put_strike': self.long_put['strike'],
            'short_put_delta': self.short_put['delta'],
            'long_put_delta': self.long_put['delta'],
            'call_spread': self.call_spread,
            'put_spread': self.put_spread,
            'net_premium': self.net_premium
        }
    
def get_spx_closing_prices(train_dates, ic_df):
    # converting 20250708 to '2025-07-08' to get spx price data from yfinance
    start = train_dates[0][0:4] + "-" + train_dates[0][4:6] + "-" + train_dates[0][6:]

    # yfinance end date is exclusive, so adding 1 business day
    last_expiration = ic_df['expiration_date'].max()
    end_date = pd.to_datetime(last_expiration, format='%Y%m%d') + (1 * US_business_days)
    end = end_date.strftime('%Y-%m-%d')

    spx = yf.download('^SPX', start=start, end=end, auto_adjust=True, progress=False)
    spx_close = spx['Close']
    spx_close.index = spx_close.index.strftime('%Y%m%d')

    return spx_close

def calculate_pnl(row):
    end_price = row['closing_price_at_expiration']
    short_call_strike = row['short_call_strike']
    long_call_strike = row['long_call_strike']
    short_put_strike = row['short_put_strike']
    long_put_strike = row['long_put_strike']   
    call_spread = row['call_spread']
    put_spread = row['put_spread']
    net_premium = row['net_premium']

    call_pnl = 0
    if end_price > long_call_strike:
        call_pnl = call_spread
    elif end_price > short_call_strike and end_price < long_call_strike:
        call_pnl = end_price - short_call_strike
    else:
        call_pnl = 0

    put_pnl = 0
    if end_price > short_put_strike:
        put_pnl = 0
    elif end_price < short_put_strike and end_price > long_put_strike:
        put_pnl = short_put_strike - end_price
    else:
        put_pnl = put_spread

    return net_premium - (call_pnl + put_pnl)

def display_df(ic_df):
    ic_df.to_html('results.html', index=False)
    full_path = 'file://' + os.path.abspath('results.html')
    webbrowser.open(full_path)

def get_position_mtm_value(ic_row, day):
    day = day.strftime('%Y%m%d')
    path = f'/Volumes/SSD/SPX_minute_split_2019-2020/{day}/160000.csv.zst'
    try:
        today_data = pd.read_csv(path, compression='zstd')
    except FileNotFoundError:
        return 0
    
    expiration = ic_row['expiration_date'].strftime('%Y%m%d')

    trade_data = today_data[today_data['expiration'] == int(expiration)]
    if trade_data.empty:
        return 0
    
    short_calls = trade_data[(trade_data['option_type'] == 'C') & (trade_data['strike'] == ic_row['short_call_strike'])]
    long_calls  = trade_data[(trade_data['option_type'] == 'C') & (trade_data['strike'] == ic_row['long_call_strike'])]
    short_puts  = trade_data[(trade_data['option_type'] == 'P') & (trade_data['strike'] == ic_row['short_put_strike'])]
    long_puts   = trade_data[(trade_data['option_type'] == 'P') & (trade_data['strike'] == ic_row['long_put_strike'])]

    if(short_calls.empty or long_calls.empty or short_puts.empty or long_puts.empty):
        # print("Could not find short/long call/put for MTM")
        return 0

    short_call_asks = short_calls[short_calls['ask'] > 0]['ask'] # buy back short
    long_call_bids  = long_calls[long_calls['bid'] > 0]['bid'] # sell long
    short_put_asks  = short_puts[short_puts['ask'] > 0]['ask']
    long_put_bids  = long_puts[long_puts['bid'] > 0]['bid']
    
    if(short_call_asks.empty or long_call_bids.empty or short_put_asks.empty or long_put_bids.empty):
        # print("Could not find asks/bids for MTM")
        return 0
    
    short_call_cost = short_call_asks.min()
    long_call_value  = long_call_bids.max()
    short_put_cost  = short_put_asks.min()
    long_put_value   = long_put_bids.max()

    closing_value = (long_call_value + long_put_value) - (short_call_cost + short_put_cost)
    unrealized_pnl = (ic_row['net_premium'] - closing_value)*100
    return unrealized_pnl

def simulate_portfolio(ic_df, initial_capital=INITIAL_CAPITAL):
    portfolio_start = ic_df['entry_date'].min()
    # portfolio_end = ic_df['entry_date'].max() # dont have data for dates after the last entry
    portfolio_end = ic_df['expiration_date'].max()
    trading_days = pd.date_range(portfolio_start, portfolio_end, freq=US_business_days)

    entries = dict(zip(ic_df['entry_date'].dt.normalize(), ic_df.index))
    expiries = dict(zip(ic_df['expiration_date'].dt.normalize(), ic_df.index))

    cash = initial_capital
    portfolio_history = []
    open_trades = set()

    portfolio_history.append({
        'date': portfolio_start- (1*US_business_days), # day before we make first trade
        'cash': initial_capital,
        'open_trades': 0,
        'margin_locked': 0,
        'unrealized_pnl': 0
    })

    for day in trading_days:
        day = day.normalize()
        if day in expiries:
            idx = expiries[day]
            if idx in open_trades:
                cash += ic_df['max_loss'][idx]
                pnl = ic_df['pnl_dollars'][idx] - (ic_df['net_premium'][idx] * 100) # already added net premium in original p&l calculation
                cash += pnl
                open_trades.remove(idx)
            
        if day in entries:
            idx = entries[day]
            if cash >= ic_df['max_loss'][idx]:
                open_trades.add(idx)
                cash -= ic_df['max_loss'][idx]
                cash += ic_df['net_premium'][idx]*100
                ic_df.at[idx, 'taken'] = True
        
        mtm_value = 0
        for idx in open_trades:
            mtm_value += get_position_mtm_value(ic_df.iloc[idx], day)
    
        portfolio_history.append({
            'date': day,
            'cash': cash,
            'open_trades': len(open_trades),
            'margin_locked': sum(ic_df['max_loss'].iloc[list(open_trades)]) if open_trades else 0,
            'unrealized_pnl': mtm_value
        })
    
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['equity'] = round((portfolio_df['cash'] + portfolio_df['margin_locked'] + portfolio_df['unrealized_pnl']),2)
    return portfolio_df

def compute_metrics(portfolio_df):
    df = portfolio_df.copy()
    df = df.set_index('date')
    df['returns'] = df['equity'].pct_change().fillna(0) # daily returns

    trading_days_per_year = 252

    total_return = (df['equity'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    num_years = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days / 365.25
    avg_annual_returns = ((portfolio_df['equity'].iloc[-1] / INITIAL_CAPITAL) ** (1/num_years)) - 1

    sharpe_ratio = (df['returns'].mean() / df['returns'].std()) * (trading_days_per_year ** 0.5) # assuming Rf = 0

    return {
        "Total Returns": round(total_return*100, 2),
        "Annualized Returns": round(avg_annual_returns*100, 2),
        "Sharpe Ratio": round(sharpe_ratio,2)
    }

def graph_portfolio(portfolio_df):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df['date'], portfolio_df['equity'], linewidth=2, color='black')
    plt.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Equity (DTE: {DTE}, DELTAS: CALL({CALL_DELTA_TARGET}), PUT({PUT_DELTA_TARGET}))')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('portfolio_returns.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    data_directory = '/Volumes/SSD/SPX_minute_split_2019-2020'
    dates = sorted([d for d in os.listdir(data_directory) if d.isdigit()])

    last_date = pd.to_datetime(dates[-1], format='%Y%m%d')
    cutoff_date = last_date - (DTE * US_business_days) # only make trade if we have data for the future (for MTM calculations)

    dates = [d for d in dates if pd.to_datetime(d, format='%Y%m%d') <= cutoff_date]

    iron_condors = [] 
    for entry_date in dates:
        df = pd.read_csv(f'{data_directory}/{entry_date}/{ENTRY_TIME}.csv.zst', compression='zstd')
        
        target_date = add_trading_days(entry_date, DTE)
        filtered_df = df[df['expiration'] == int(target_date)] 

        long_call, short_call = find_call_options(filtered_df)
        long_put, short_put = find_put_options(filtered_df)
        if long_call is None or long_put is None:
            continue
    
        ic = IronCondor(entry_date, target_date, long_call, short_call, long_put, short_put)
        iron_condors.append(ic)
    
    ic_df = pd.DataFrame([ic.to_dict() for ic in iron_condors])

    spx_close = get_spx_closing_prices(dates, ic_df)
    ic_df = ic_df.merge(spx_close, left_on='expiration_date', right_index=True, how='left')
    ic_df = ic_df.rename(columns={'^SPX' : 'closing_price_at_expiration'})

    ic_df['pnl_dollars'] = round(ic_df.apply(calculate_pnl, axis=1)*100,2)
    ic_df['entry_date'] = pd.to_datetime(ic_df['entry_date'], format='%Y%m%d')
    ic_df['expiration_date'] = pd.to_datetime(ic_df['expiration_date'], format='%Y%m%d')
    print("="*50)
    print(f"TOTAL P&L: {ic_df['pnl_dollars'].sum()}")
    # display_df(ic_df)

    ic_df['max_loss'] = (ic_df[['call_spread', 'put_spread']].max(axis=1) - ic_df['net_premium'])*100
    ic_df['taken'] = False
    portfolio_df = simulate_portfolio(ic_df)

    metrics = compute_metrics(portfolio_df)
    for k,v in metrics.items():
        print(f"{k}: {v}")

    graph_portfolio(portfolio_df)