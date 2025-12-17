import numpy as np

TRIALS = 100000

def gen_spherical_coords(radius = 1.0):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(np.random.uniform(-1, 1))
    return [radius, theta, phi]

def angular_dist_sphere(A, B):
    distance = np.arccos(np.sin(A[2])*np.sin(B[2])*np.cos(A[1]-B[1]) + np.cos(A[2])*np.cos(B[2]))
    return distance

if __name__ == "__main__":
    true_cnt_sphere = 0.0
    for i in range(TRIALS):
        A = gen_spherical_coords()
        B = gen_spherical_coords()
        C = gen_spherical_coords()

        if(angular_dist_sphere(A,B) < np.pi/2 and angular_dist_sphere(A,C) < np.pi/2) or \
        (angular_dist_sphere(B,A) < np.pi/2 and angular_dist_sphere(B,C) < np.pi/2) or \
        (angular_dist_sphere(C,A) < np.pi/2 and angular_dist_sphere(C,B) < np.pi/2):
            true_cnt_sphere+=1

    print(true_cnt_sphere/TRIALS)
