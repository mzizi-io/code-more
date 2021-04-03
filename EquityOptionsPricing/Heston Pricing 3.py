#!/usr/bin/env python
# coding: utf-8

# <img alt = 'code-more' src = 'img/img.png'>
# 
# <h1><center>EQUITY OPTIONS PRICING 3</center></h1>
# 
# <h2><center>Heston Model: Monte Carlo Pricing</center></h2>
# 

# In this section, we review pricing of complex/exotic options using a Monte Carlo implementation of the Heston model.
# 
# I'll use the simplest scheme available but it should serve as a template to more advanced schemes. The idea is to get the ball rolling and use what we do here as a template for more efficient schemes.
# 
# This section is a key component of options pricing. Many of the options today are in fact priced using Monte Carlo. This is because there are no clean pricing functions that exist for more complex options. 
# 
# <h3><center>Table of contents</center></h3>

# 1. [Black Scholes SDE](#Black-Scholes-SDE)
# 2. [Heston SDE](#Heston-SDE)
# 3. [Discretization Schemes](#Discretization-Schemes)
#     - [Euler Scheme](#Euler-Scheme)

# <br>
# <br>
# <br>
# <h3><center>Black Scholes SDE</center></h3>
# 
# The assumption in the Black Scholes model is that the underlying $S_t$ is driven by the stochastic differential equation:
# 
# $$dS_t = \mu(S_t, t)dt + \sigma(S_t, t) dW_t$$
# 
# In very basic terms, you can think of $dS_t$ as the derivative of $S_t$. This means that the changes in $S_t$ are governed by the equation above.
# 
# The $\mu(S_t, t)dt$ is dependent on dt which is the change in time. That is $\mu(S_t, t)$ can be a function or a number that changes with $t$ and that we know and whose output we are certain of. This is called the **deterministic part** of this equation. 
# 
# To make this even more explicit, the $\mu(S_t, t)$ function is actually the **risk free rate** we defined as r multiplied by $S_t$. So essentially we can replace $\mu(S_t, t)$ with $rS_t$ to have a bit of simplicity.
# 
# On the other hand, $\sigma(S_t, t) dW_t$ is dependent on $dW_t$. In very simplified terms, $dW_t$ is a gaussian distribution whose mean and variance change with every step. Also,  there is no relation between $t$  and $t+dt$. This means that the gaussians are **independent across time.**
# 
# Note that we can also define $\sigma(S_t, t)$ as $\sigma.S_t$ in the Black Scholes model.
# 
# So in reality, our equation is defined as:
# 
# $$S_{t+dt} = S_t + \int_t^{t+dt}r.S_udu + \int_t^{t+dt}\sigma.S_u dW_u$$
# 
# You can see from this that the $S_{t+dt} - S_t$ is essentially the first representation in an integral.
# 

# <br>
# <br>
# <br>
# <h3><center>Heston SDE</center></h3>
# 
# If you have worked with the Black Scholes model, you know that the implicit volatility is a key component in pricing options. However, under the Black Scholes model the assumption is made that this volatility moves in tandem with the stock price and does not have variations caused by other market effects.(You can see that $\sigma(S_t, t)$ is just a value $\sigma$ multiplied by $S_t$.)
# 
# However, we know that volatility actually changes depending on market conditions.
# 
# This is the primary premise behind the Heston model. The assumption is made that the volatility moves in a different fashion from the market. Typically, what we see in equities is that the **volatility of options increases as prices decrease and vice versa.**
# 
# This means that the prices and volatilities are **negatively correlated.** You can observe this by heuristically by comparing the VIX to the S&P 500. The highest points on the VIX are the 2008 crisis and the 2020 Covid pandemic.
# 
# So if this is the case, then the $\sigma(S_t,t)$ function is not deterministic. It is in fact random and we can therefore assign another stochastic differential equation to the volatility to have two equations.
# 
# We begin by noting that instead of having an equation on $\sigma(S_t, t)$, we can have an equation on $\sigma(S_t, t)^2 = \sigma.\alpha(S_t, t)$ which we can equate intuitively to the variance.
# 
# So our final equations will be:
# 
# $$dS_t = r.S_tdt + \sigma.\sqrt{\alpha(S_t, t)} dW_t^1$$
# 
# $$d\alpha_t = \kappa(\theta - \alpha_t(S_t, t))dt + \nu\sqrt{\alpha(S_t, t)} dW_t^2$$
# 
# and because there's a relationship between the volatility and the underlying, we can set a correlation parameter between the two equations. This can only be done between the $dW_t^1$ and $dW_t^2$ because they're the only sources of randomness in our model. As we said, the other parts are deterministic.
# 
# Therefore:
# 
# $$E[dW^1_t dW^2_t] = \rho$$
# 
# And here we see our parameters appear: $\{\sigma, \theta, \kappa, \nu, \rho\}$

# <br>
# <br>
# <h4><center>S&P vs VIX</center></h4>

# In[204]:


import investpy 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get S&P and VIX quotes from investing
SP500 = investpy.get_index_historical_data(index='S&P 500',
                                   country='united states',
                                  from_date = '01/01/2000',
                                  to_date = '01/01/2021')['Close']

VIX = investpy.get_index_historical_data(index='S&P 500 VIX',
                                   country='united states',
                                  from_date = '01/01/2000',
                                  to_date = '01/01/2021')['Close']

SP500 = SP500.reset_index()
VIX = VIX.reset_index()

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles = ['S&P 500', 'VIX'])

fig.add_trace(
    go.Scatter(x=SP500['Date'], 
               y=SP500['Close'], 
               name = "S&P 500",
               line = dict(color='#ffc93c', width=2.5)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=VIX['Date'],
               y=VIX['Close'], 
               name = "VIX",
               line = dict(width=2.5)),
    row=1, col=2
)

fig.update_layout(height=500, 
                  width=1000, 
                  title_text="S&P 500 vs VIX", 
                  title_x = 0.5,
                  plot_bgcolor='#0f4c75')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# <br>
# <br>
# <br>
# <h3><center>Discretization Schemes</center></h3>

# In this section, we implement the integral SDE in a discrete form. Much like we did with the Heston probability function, we intend to break down the integral into manageable discrete pieces that we can sum together to obtain the modeled $S_T$, i.e. the modeled underlying level at the expiry of the option.
# 
# Because of the randomness produced by the two $dW_t$, we will generate a large number of trajectories and calculate the mean. This mean should represent our expectation of the future price of the underlying. This reasoning can be intuited from the pricing formulas where we have probabilities as elements of the pricing function.
# 
# 
# <h4><center>Monte Carlo Pricing Intuitively</center></h4>
# 
# We start with the payoff of a call option. At expiry, we know that the call will either give us 0 if we're below the strike or $S_T - K$. Here $T$ represents the maturity. So in effect, the value of our option today is the expected value of $max(S_T - K, 0)$ discounted by the risk free rate because we're in a risk neutral setup.
# 
# So our call should be worth: $e^{rT}E[max(S_T - K, 0)]$ where $E$ is just the sign for the expectation.
# 
# But regardless of the probability, if $S_T - K$ is negative, our option is worth 0. So we can just focus on the scenario where $S_T - K$. This leaves us with $e^{rT}E[S_T - K]$ and we already know $K$ so our option, heuristically should be worth $e^{rT}(E[S_T] -  K)$
# 
# Now if we generaate enough random $S_T$ using our two random equations, the mean of all these values should give us $E[S_T]$. The details and proofs come after the intuition but this is the general framework.
# 
# So for a complex payoff, we can replace the $max$ function with a generic $f$ and try and find  $e^{rT}E[f(S_T;K)]$ by essentially simulating possible values of $S_T$ and applying the function $f(S_T;K)$ to each scenario created.
# 
# The mean of these should give us an approximation of the risk neutral value of the option.
# 
# 

# <br>
# <br>
# <br>
# <h3><center>Euler Scheme</center></h3>
# 
# We start off with our 2 equations in integral form:
# 
# $$S_{t+dt} = S_t + \int_t^{t+dt}r.S_udu + \int_t^{t+dt}\sigma.\sqrt{\alpha(S_u, u)}S_u dW_u^1$$
# 
# $$\alpha_{t+dt} = \alpha_t + \int_t^{t+dt}\theta(\kappa - \alpha_u)du + \int_t^{t+dt}\nu.\sqrt{\alpha_u}.dW_u^2$$
# 
# Because the first equation needs the $\alpha$ as an input, we start by simulating the $\alpha$ before filling the value we obtain into the equation of $S_{t+dt}$
# 
# Because $dt$ is so small, we can essentially approximate $\int_t^{t+dt}\theta(\kappa - \alpha_u)du$ to be $\theta(\kappa - \alpha_u)dt$.
# 
# Similarly, because of some special properties of $dW_t$, we can approximate the second integral $\int_t^{t+dt}\nu.\sqrt{\alpha_u}.dW_u^2$  as $\sqrt{\alpha_t}(dW_{t+dt} - dW_t)$.
# 
# We can see that the leap is not too big. However, providing these proofs is far from trivial.
# 
# Because of $dW_t$ one again, we can show $dW_{t+dt} - dW_t$ to be a gaussian distribution of mean 0 and variance $dt$. 
# 
# So in essence $ \int_t^{t+dt}\nu.\sqrt{\alpha_u}.dW_u^2 \approx \nu\sqrt{\alpha_t}dW_{t+dt} - dW_t = \nu\sqrt{\alpha_t}Z\sqrt{dt}$ where Z is just the normal distribution.
# 
# <h4>Approximation One</h4>
# 
# So our first approximation is:
# 
# $$\alpha_{t+dt} = \alpha_{t} + \kappa(\theta - \alpha_t)dt + \nu\sqrt{\alpha_t.dt}Z_t$$
# 
# <h4>Approximation Two</h4>
# 
# Using the same logic as above we should obtain:
# $$S_{t+dt} = S_t + rS_tdt + \sigma\sqrt{\alpha_tdt}S_tZ_t$$
# 
# 
# <br>
# <br>
# <br>
# <h3><center>Implementation</center></h3>
# To implement this, we will do the following:
# 
# 1. Determine the number of time steps we want in each year eg dt could be daily,weekly, monthly or yearly 
# 2. Determine the number of iterations we want e.g we could generate $S_T$ 1000, 10000 or even a million times
# 3. Declare all the gaussians we will use beforehand. This helps speed up the code.
#     - In this case, we have X number of timesteps which is our timesteps x maturity and Y the number of iterations
#     - So we will declare 2 gaussians of size (X, Y) and we will correlate them by $\rho$
#     - To do this, we use the [Cholesky decomposition](#https://en.wikipedia.org/wiki/Cholesky_decomposition)
# 4. For each timestep, we calculate all $v_t$ at the same time. The numpy package allows us to do vectorized calculations quickly.
# 5. We use all the $v_t$ we obtain to calculate $S_t$
# 6. We subtract the mean of all $S_T$ from $K$ and discount it at the rate of $e^{rT}$ to get the present value of our option.

# In[357]:


# implementation of MC
import numpy as np
def MCHeston(St, K, r, T, sigma, kappa, theta, volvol, rho, iterations, timeStepsPerYear):
    timeStepsPerYear = 12
    iterations = 1000000
    timesteps = T * timeStepsPerYear
    dt = 1/timeStepsPerYear

    # Define the containers to hold values of St and Vt
    S_t = np.zeros((timesteps, iterations))
    V_t = np.zeros((timesteps, iterations))

    # Assign first value of all Vt to sigma
    V_t[0,:] = sigma
    S_t[0, :] = St

    # Use Cholesky decomposition to
    means = [0,0]
    stdevs = [1/3, 1/3]
    covs = [[stdevs[0]**2          , stdevs[0]*stdevs[1]*rho], 
            [stdevs[0]*stdevs[1]*rho,           stdevs[1]**2]] 

    Z = np.random.multivariate_normal(means, covs, (iterations, timesteps)).T
    Z1 = Z[0]
    Z2 = Z[1]

    for i in range(1, timesteps):
        # Use Z2 to calculate Vt
        V_t[i,:] = np.maximum(V_t[i-1,:] + kappa * (theta - V_t[i-1,:])* dt + volvol * np.sqrt(V_t[i-1,:] * dt) * Z2[i,:],0)
        
        # Use all V_t calculated to find the value of S_t
        S_t[i,:] = S_t[i-1,:] + r * S_t[i,:] * dt + np.sqrt(V_t[i,:] * dt) * S_t[i-1,:] * Z1[i,:]  

    return np.mean(S_t[timesteps-1, :]- K)


# <img alt = 'code-more' src = 'img/img.png'>
