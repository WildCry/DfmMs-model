import numpy as np
import matplotlib.pyplot as plt
import kalman


plt.ion()
plt.figure()

kf = kalman.KalmanFilter(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

DT = 0.1
NUM_STEPS = 1000
MEAS_EVERY_STEPS = 20

real_x = 0.0
meas_variance = 0.1 ** 2
real_v = 0.9

mus = []
covs = []
real_xs = []
real_vs = []

for step in range(NUM_STEPS):
    if step > 500:
        real_v *= 0.9
    covs.append(kf.cov)
    mus.append(kf.mean)

    real_x = real_x + DT * real_v

    kf.predict(dt=DT)

    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        kf.update(meas_value=real_x + np.random.randn() * np.sqrt(meas_variance),
                  meas_variance=meas_variance)

    real_xs.append(real_x)
    real_vs.append(real_v)


plt.subplot(2, 1, 1)
plt.title('Position')
plt.plot([mu[0] for mu in mus], c='r')
plt.plot(real_xs, c='b')
plt.plot([mu[0] - 2*np.sqrt(cov[0, 0])
         for mu, cov in zip(mus, covs)], c='r', linestyle='--')
plt.plot([mu[0] + 2*np.sqrt(cov[0, 0])
         for mu, cov in zip(mus, covs)], c='r', linestyle='--')

plt.subplot(2, 1, 2)
plt.title('Velocity')
plt.plot([mu[1] for mu in mus], c='r')
plt.plot(real_vs, c='b')
plt.plot([mu[1] - 2*np.sqrt(cov[1, 1])
         for mu, cov in zip(mus, covs)], c='r', linestyle='--')
plt.plot([mu[1] + 2*np.sqrt(cov[1, 1])
         for mu, cov in zip(mus, covs)], c='r', linestyle='--')

plt.show()
plt.ginput(1)
