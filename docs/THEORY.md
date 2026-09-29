# Theory

The derivations behind the three methods, kept from the original write-up of this
project and extended where the implementation has since gone further.

For what the code actually does with these, see the [README](../README.md); for the
numbers it produces, [VALIDATION.md](VALIDATION.md).

> **Note on the finite-difference section.** The original version of this document
> described the Crank-Nicolson discretisation but reported that the implementation
> had "blown up values" and had been removed. That is no longer the case. The scheme
> derived below is implemented in
> [`src/options_pricing/crank_nicolson.py`](../src/options_pricing/crank_nicolson.py)
> and converges at its theoretical second order (measured: 2.00). The diagnosis in
> that section — that the instability came from using the *explicit* scheme with
> large time steps — was correct: the explicit method is stable only while
> `dt <= dS^2 / (sigma^2 S_max^2)`, which a grid fine enough to resolve the payoff
> kink makes prohibitive. Crank-Nicolson averages the explicit and implicit schemes
> and is unconditionally stable for this equation.

---

# Stock Price Process to Black Scholes 
Using the stock process typically defined, we start with:

$$dS = \mu S dt + \sigma S dz$$

### Drift Rates
where we have the $\mu S$ defined as the drift rate since we take S as the price and the parameter $\mu$ as the expected rate of return on a stock. We can easily see that without the uncertainty and variation term $\mu S dt$ we would get $\int_{0}^{T}S_T = S_0e^{\mu T}$ as a compounding of the drift rate through time T, or time to maturity.

### Volatility/Uncertainty
Our assumption, is that the variability is constant over each of the short time periods($\Delta t$) s.t. we can utilize $\sigma$ throughout the model. This follows geometric brownian motion in a discrete time model as we are operating on small intervals of $\Delta t$.

So, with that we can actually further our understanding of the process using a **call** or **put**.

## Using Ito's Lemma for the price

Let *f* represent the price of a call, where *f* is a function of S and t.

$$df = \left( \frac{\partial{f}}{\partial{S}} \mu S + \frac{\partial{f}}{\partial{t}} + \frac{1}{2} \frac{\partial{f^2}}{\partial^2{S}} \right)dt + \frac{\partial{f}}{\partial{S}} \sigma S dz$$

Using the discrete values we have $\Delta S = \mu S \Delta t + \sigma S \Delta z$. Additionally, $Delta z = \epsilon \sqrt{ \Delta t}$

which we can use to reformat *f* as:

$$\Delta f = \left( \frac{\partial{f}}{\partial{S}} \mu S + \frac{\partial{f}}{\partial{t}} + \frac{1}{2} \frac{\partial{f^2}}{\partial^2{S}} \right) \Delta t + \frac{\partial{f}}{\partial{S}} \sigma S \epsilon \sqrt{\Delta t}$$

## Constructing the option

We define the portfolio as $\Pi$ and use:

$$\Pi = -f + \frac{\partial{f}}{\partial{S}}S$$

as our portfolio. In this form, we represent the *-f* as the short position of a derivative and  $\frac{\partial{f}}{\partial{S}}$ as a long position on shares of the underlying. If we then allow for discrete time steps we get the $\Delta \Pi$ format which enables us to input $\Delta f$ for the function. Taking into account the assumptions of the model that the values will follow the near term riskless security the function collapses to a simpler $\Delta \Pi = r \Pi \Delta t$. Since we have $ \Delta \Pi$ as a formula and $ \Pi$ as a formula, we can insert them as follows:

$$\left(\frac{\partial f}{\partial S} + 0.5 \cdot \frac{\partial^2 f}{\partial S^2} \sigma^2 \S^2 \right) \Delta t = r \left(f - \frac{\partial f}{\partial S} \right) \Delta t \Rightarrow \frac{\partial f}{\partial t} + r S \frac{\partial f}{\partial S} + 0.5 \cdot \sigma^2 S^2 \frac{\partial^2 f}{\partial S^2} - rf = 0$$

Our value is $f = \text{max}(S-K,0) \text{\quad when t = T}$, which results in our option.

# Finite Difference

*Originally titled "Attempted/Reworking". Now implemented; see the note above.*

My primary error was in the conversion to a diffusion process of the values. When I worked on this project for classes, I tried to do so for the issues related to boundary problems, but I didn't understand the Black-Scholes equation's clutter caused issues in converging to a solution. While it would be easier to transition, I do want to try out another form. Since my initial attempt at the Finite Difference method was the explicit, rather than the implicit method. Since I had rather large time steps, reviewing numerical procedures later with the Options, Futures, and Other Derivatives text made me realize my likely instability of the model was due to the combination of the complexity from early exercise in american options and the lesser stability of the explicit method.  

## Crank-Nicolson Finite Difference

In the first model I used, I had semi-trustworthy results, but not robust results. This is likely due to my usage of too large $\Delta t$ combined with the explicit finite difference method. The advantage of the Crank-Nicolson is its position as an average of the implicit and explicit finite difference method. The first issue in solving this problem is to shift the BSM model into a discretized form. The main way of doing this is to use a 1/2 fraction of the prior and current values with respect to price where our central approx for the first term is $ f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$ and the standard approx for *f"(x)* which are approximated by:

$$\frac{f_{i-1/2, j}}{\partial S} = 0.5 \cdot \left[\frac{\partial f_{i-1,j} + \partial f_{i,j}}{\partial S} + \frac{}{\partial S} \right]$$

which is expanded as $= 0.5 \cdot \left[ \frac{\partial f_{i-1,j} + \partial f_{i,j}}{2 \delta S} + \frac{f_{i, j+1} - f+{i,j-1}}{2 \delta S} \right] + O( \delta S^2)$.

Our second term is approximated by: 

$$\frac{\partial^2 f_{i-1/2,j}}{\partial S^2} = 0.5 \cdot \left[ \frac{\partial^2 f_{i-1,j}}{\partial S^2} + \frac{\partial^2 f_{i,j}}{\partial S^2}\right]$$

which is expanded as $= 0.5 \cdot \left[ \frac{f_{i-1,j+1} - 2 f_{i-1,j} + f_{i-1,j-1}}{\partial S^2} + \frac{f_{i,j+1} - 2 f_{i,j} + f_{i,j-1}}{\partial S^2}  \right] + O( \delta S^2)$.

These values can be implemented into the BSM PDE and collected to get: $a_j f_{i, j-1} + (1-b_j) f_{i,j} + c_j f_{i, j+1}$

where:

$$ a_j = \frac{\delta t}{4} (\sigma ^2 j^2 - rj)$$

$$ b_j = - \frac{\delta t}{2} (\sigma ^2 j^2 + r)$$

$$ c_j = \frac{\delta t}{4} (\sigma ^2 j^2 + rj)$$

The *i* and *j* values represent values on a lattice from values between today and expiration and the price into N levels. We can solve the for f at each of the nodes of the lattice by solving the set of simultaneous equations above. (This is typically done in a matrix formation). Using the stability conditions for this method, where the values need to be under normal conditions, there is stability into infinity from the ratio of the prior and current period. The $O(\delta t^2$ and  $O(\delta S^2)$ are the convergence rates of the Crank-Nicolson Method. For greeks, the Crank-Nicolson value seems to have occasional issues, and often has been supplemented by taking a small set of periods with the explicit method before switching to Crank-Nicolson. 
