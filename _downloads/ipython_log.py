# IPython log file

def simple_pendulum(theta_thetadot, t):
        theta, theta_dot = theta_thetadot
        return [theta_dot, - np.sin(theta)]
from scipy import odeint
from scipy.integrate import odeint
theta_thetadot = (np.pi / 3, 0)
t = np.linspace(0, 5 * np.pi, 1000)
sol = odeint(simple_pendulum, (np.pi/3, 0), t)
theta, theta_dot = sol.T
plot(t, theta_dot)
plot(t, theta)
clf()
plot(t, theta, lw=3)
plot(t, theta_dot, lw=3)
clf()
plot(t, theta_dot, lw=3, label=u'$\theta$')
legend()
clf()
plot(t, theta_dot, lw=3, label=u'$\\theta$')
legend()
get_ipython().magic(u'logon')
get_ipython().magic(u'logstart')
get_ipython().magic(u'logstop')
