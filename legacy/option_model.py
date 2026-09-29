"""SUPERSEDED. Kept for reference only -- this file is not part of the package and
is known to be incorrect.

This is the original coursework implementation. It is preserved because the README's
"What was wrong" section refers to it, and because the defects in it are more
instructive read against the replacement than described in prose. Do not import it.

Known defects, each with a named regression test in ``tests/test_numerical.py``:

1. ``p = np.exp((self.r-self.q)*(self.t/self.N) - down_factor)/(up_factor-down_factor)``
   subtracts the down-factor inside the exponential. Gives p = 6.69 at 50 steps and
   13.19 at 200, so the price diverges as steps are added.
2. ``exercise = np.maximum(0, price_tree[self.t, i] - self.K ...)`` indexes the lattice
   at the period argument instead of the backward-induction level ``n``.
3. Put rho is returned positive; it must be negative.
4. Call vega carries a spurious ``1/np.sqrt(2*math.pi)`` that put vega does not.
5. ``dt = self.t/self.N`` conflates the period index with time to maturity, and
   ``N - t`` is used as both a year count and a step count.
6. Pricing calls ``plt.show()`` from inside the pricing routine and ends in
   ``return print(...)``, so it returns ``None``.

The working implementation is the flat modules at the repository root:
``black_scholes.py``, ``binomial.py`` and ``crank_nicolson.py``.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math



class Options:
    def __init__(self, S, K, t, N, r, sigma, option_type, q = 0, exercise_type='euro'):
      """
      S = Spot price at any period t
      K = Strike Price
      t = period of spot price
      N = total periods(time to maturity is N - t)
      r = interest rate (Risk Free)
      sigma = volatility
      option_type = call or put
      q = dividends
      exercise_type = euro or amer
      dt = ratio of step to time to maturity
      """
      self.exercise_type = exercise_type
      self.S = S
      self.K = K
      self.t = t
      self.N = N
      self.r = r
      self.sigma = sigma
      self.option_type = option_type
      self.q = q
      self.exercise_type = exercise_type
      #self.dt = t/N





    def euro_price(self):
            """
            d1:The value is assessed by taking current stock prices and multiplying them by a probability factor (d1)

            d2: Then subtracts the discounted exercise payment by a second probability factor (d2)
            sigma measures the volaitily of the stock

            The following Greeks for a derivative are the most commonly used option Greeks.
            All Greeks are first or second-order partial derivatives of the option price in reference to additional factors.
            Greeks are used to measure the responsiveness of price to the small change prices of each parameter.

            vega: vega measures sensitivity to volatility. Vega is the derivative of the option value with respect to the volatility of the underlying asset

            gamma: Gamma measures the rate of change in the delta with respect to changes in the underlying price.

            rho: Rho measures the sensitivity to the interest rate.

            delta: Delta measures the rate of change of the theoretical option value with respect to changes in the underlying asset’s price.

            theta: Theta measures the sensitivity of the value of the derivative to the passage of time.
            """



            # d1 takes the ln of the ratio of the S to K + the time period multiplied by the interest rate less a dividend * 1/2(sigma ^2)
            # dividend variable was denoted as q due to d variable already taken. This is one of the several common variables used in the literature to denote dividend.


            d1 = (np.log(self.S/self.K) + (self.N-self.t)*(self.r - self.q + (0.5*self.sigma**2)))/ (self.sigma * np.sqrt(self.N - self.t))

            d2  = d1 - self.sigma * np.sqrt(self.N-self.t)

            # Establishes lists to enable plotting of greeks.


            Value = []
            Vega = []
            Gamma = []
            Rho = []
            Delta = []
            Theta = []


            # Separation of call/put calculations.


            if self.option_type == "call":
                # Sets up movement of the first and second derivatives of the model along with the relative changing of all greek values.
                for i in range(self.t, self.N):
                  # identified the same as d1 and d2 above, substituting t for i as an iterable.
                  d1 = (np.log(self.S/self.K) + (self.N-i)*(self.r - self.q + (0.5*self.sigma**2)))/ (self.sigma * np.sqrt(self.N - i))

                  d2  = d1 - self.sigma * np.sqrt(self.N-i)

                  # value calculation for the price based on each time period
                  value = self.S * np.exp(-self.q*(self.N-i)) * norm.cdf(d1) - (self.K * np.exp(-self.r * (self.N-i)) * norm.cdf(d2))

                  # Vega was defined earlier, it is one of the financial derivatives of BSM model.
                  vega = self.S * np.exp(-self.q*(self.N-self.t)) * np.sqrt(self.N-self.t) * 1/(np.sqrt(2*math.pi)) * np.exp(-d1**2/2)

                  # Gamma was defined earlier, it is one of the financial derivatives of BSM model.
                  gamma = (np.exp(-self.q * (self.N - self.t)))/(self.S * self.sigma * np.sqrt(self.N-self.t)) * np.exp(-d1**2/2)

                  # Rho was defined earlier, it is one of the financial derivatives of BSM model.
                  rho = (self.K * (self.N-i) * np.exp(-self.r * (self.N - i)) * norm.cdf(d2))

                  # Delta was defined earlier, it is one of the financial derivatives of BSM model.
                  delta = np.exp(-self.q * (self.N-self.t))*norm.cdf(d1)

                  # Theta was defined earlier, it is one of the financial derivatives of BSM model.
                  theta = - (self.S * self.sigma * np.exp(-self.q * (self.N-self.t)))/(2 * np.sqrt(2 * math.pi * (self.N - self.t))) * 1/(np.sqrt(2 * math.pi)) * np.exp(-(d1**2)/2) + self.q * self.S * np.exp(-self.q * (self.N - self.t)) * norm.cdf(d1) - self.r * self.K * np.exp(-self.r * (self.N - self.t)) * norm.cdf(d2)

                  # Appending values to previous defined lists
                  Value.append(value)
                  Vega.append(vega)
                  Gamma.append(gamma)
                  Rho.append(rho)
                  Delta.append(delta)
                  Theta.append(theta)



                # The following code enables plotting of the greeks over time. We were originally planning a 3D model, using x as time, y as underlying asset value, and z as the greek.
                # We ran into trouble plotting them as we had case issues with the plotting by matplotlib
                plt.plot(Value)
                plt.xlabel('Time')
                plt.ylabel('Option Value')
                plt.title('European Option Value Over Time')
                plt.show()

                plt.plot(Vega)
                plt.xlabel('Time')
                plt.ylabel('Vega')
                plt.title('Vega value Over Time')
                plt.show()

                plt.plot(Gamma)
                plt.xlabel('Time')
                plt.ylabel('Gamma')
                plt.title('Gamma Over Time')
                plt.show()

                plt.plot(Rho)
                plt.xlabel('Time')
                plt.ylabel('Rho')
                plt.title('Rho value Over Time')
                plt.show()

                plt.plot(Delta)
                plt.xlabel('Time')
                plt.ylabel('Delta')
                plt.title('Delta Over Time')
                plt.show()

                plt.plot(Theta)
                plt.xlabel('Time')
                plt.ylabel('Theta')
                plt.title('Theta value Over Time')
                plt.show()

                # returns the price of the BSM along with the ending greek derivatives.
                return print("value = ", self.S * np.exp(-self.q*(self.N-self.t)) * norm.cdf(d1) - (self.K * np.exp(-self.r * (self.N-self.t)) * norm.cdf(d2)),
                             "\nvega = ", vega,
                             "\ngamma = ", gamma,
                             "\nrho = ", rho,
                             "\ndelta = ", delta,
                             "\ntheta = ", theta)


            else:
                # Sets up movement of the first and second derivatives of the model along with the relative changing of all greek values.
                for i in range(self.t, self.N):
                  # identified the same as d1 and d2 above, substituting t for i as an iterable.
                  d1 = (np.log(self.S/self.K) + (self.N-i)*(self.r - self.q + (0.5*self.sigma**2)))/ (self.sigma * np.sqrt(self.N - i))

                  d2  = d1 - self.sigma * np.sqrt(self.N-i)

                  # value calculation for the price based on each time period
                  value = self.K * np.exp(-self.r * (self.N - i)) * norm.cdf(-d2) - self.S * np.exp(-self.q * (self.N - i)) * norm.cdf(-d1)

                  # Vega is a previously defined derivative, and is calculated exactly the same from call/put.
                  vega = self.S * np.exp(-self.q*(self.N-self.t)) * np.sqrt(self.N-self.t) * np.exp(-d1**2/2)

                  # Gamma is a previously defined derivative, and is calculated exactly the same from call/put.
                  gamma = (np.exp(-self.q * (self.N - self.t)))/(self.S * self.sigma * np.sqrt(self.N-self.t)) * np.exp(-d1**2/2)

                  # Rho is a previously defined derivative, and is calculated with only a negative e^x and a negative d2 value.
                  rho = (self.K * (self.N - i) * np.exp(-self.r * (self.N - i)) * norm.cdf(-d2))

                  # Delta is a previously defined derivative, and is calculated with only a negative e^x and a negative d1 value.
                  delta = -np.exp(-self.q*(self.N - i))*(norm.cdf(-d1))

                  # Theta is a previously defined derivative, and is calculated with some negative parameters in the function group.
                  theta = - (self.S * self.sigma * np.exp(-self.q * (self.N - i)))/(2 * np.sqrt(2 * math.pi * (self.N - i))) * 1/(np.sqrt(2 * math.pi)) * np.exp(-(d1**2)/2) - self.q * self.S * np.exp(-self.q * (self.N - i)) * norm.cdf(-d1) + self.r * self.K * np.exp(-self.r * (self.N - i)) * norm.cdf(-d2)

                  # Appending values to previous defined lists
                  Value.append(value)
                  Vega.append(vega)
                  Gamma.append(gamma)
                  Rho.append(rho)
                  Delta.append(delta)
                  Theta.append(theta)


                plt.plot(Value)
                plt.xlabel('Time')
                plt.ylabel('Option Value')
                plt.title('European Option Value Over Time')
                plt.show()


                plt.plot(Vega)
                plt.xlabel('Time')
                plt.ylabel('Vega')
                plt.title('Vega value Over Time')
                plt.show()


                plt.plot(Gamma)
                plt.xlabel('Time')
                plt.ylabel('Gamma')
                plt.title('Gamma Over Time')
                plt.show()

                plt.plot(Rho)
                plt.xlabel('Time')
                plt.ylabel('Rho')
                plt.title('Rho value Over Time')
                plt.show()

                plt.plot(Delta)
                plt.xlabel('Time')
                plt.ylabel('Delta')
                plt.title('Delta Over Time')
                plt.show()


                plt.plot(Theta)
                plt.xlabel('Time')
                plt.ylabel('Theta')
                plt.title('Theta value Over Time')
                plt.show()


                return print("value = " , self.K * np.exp(-self.r * (self.N - self.t)) * norm.cdf(-d2) - (self.S * np.exp(-self.q * (self.N-self.t)) * norm.cdf(-d1)),
                             "\nvega = ", vega,
                             "\ngamma = ", gamma,
                             "\nrho = ", rho,
                             "\ndelta = ", delta,
                             "\ntheta = ", theta)


    """
        def price_tree(self):

        Construct a binomial tree of stock prices.

        Construcs
        price_tree = [[self.S]]
        for i in range(1, self.N + 1):
            prices = [self.S * (up_factor ** j) * (down_factor ** (i - j)) for j in range(i + 1)]
            price_tree.append(prices)
        return price_tree
    """


    def binomial_american(self):
          dt = self.t/self.N
          # Up/Down Factors determining the probaibility of an up or down movement.
          up_factor = np.exp(self.sigma * np.sqrt(dt))
          down_factor = 1/up_factor

          # Creates an up down tree using initial S value applying up down calculations across rows and columns for a grid of probabilities.
          price_tree = np.zeros((self.N + 1, self.N + 1))
          price_tree[0, 0] = self.S
          for i in range(1, self.N + 1):
              price_tree[i, 0] = price_tree[i - 1, 0] * down_factor
              for j in range(1, i + 1):
                  price_tree[i, j] = price_tree[i - 1, j - 1] * up_factor

          # Calculates a risk neutral probability measure as is found in Stefanica.
          p = np.exp((self.r-self.q)*(self.t/self.N) - down_factor)/(up_factor-down_factor)

          # Determines maximum values at each period replacing the maximum values with 0 if crossing negative boundary.
          option_values = np.maximum(0, (price_tree[self.N, :] - self.K) if self.option_type == "call" else (self.K - price_tree[self.N, :]))

          # Similarly to before, applies across rows and columns a hold and exercise calculation preserving the calculations of the higher value.
          option_price = []
          for n in range(self.N-1,-1,-1):

            for i in range(n+1):

              # hold calculation, determining current price.
              hold = np.exp(-self.r * (self.t/self.N)) * (p * option_values[i + 1] + (1 - p) * option_values[i])

              # Exercise calculation, determining inequality values up the nodes.
              exercise = np.maximum(0, price_tree[self.t, i] - self.K if self.option_type == "call" else self.K - price_tree[self.t, i])

              # Returns maximum result of the two calculations.
              option_values[i] = np.maximum(hold, exercise)
              option_price.append(option_values[i])
          plt.plot(option_price)
          plt.xlabel('Time')
          plt.ylabel('Option Value')
          plt.title('Aemrican Options Value Over Time')
          plt.show()
          # Price return of ending node for hold v. exercise
          return print("Price = ", option_values[0])

    # Establishes checks against improper variable assignment, specifically ones that are not provided defaults or would be incalcuable without.
    def price(self):
        if self.exercise_type == 'euro':
          # Handling possible assignment of no time step in the option calculation
            if (self.N - self.t) == 0:

                if self.option_type == "call":
                    # If the option is a call type, we will return a max of the spot price less the strike price. No need to calculate further as there is no step period.
                    return max(self.S - self.K,0)

                elif self.option_type == "put":
                    # If the option type is a put type, returns a max of the strike price less the spot price. No need to calculate further as there is no step period.
                    return max (self.K - self.S,0)

                else:
                    # This ensures the user will input one option type. Since this is a required input type, we opted to raise an error rather than set a default value.
                    raise ValueError("You must provide an option type of either call or put. try again.")
            # Handles possibility of N being less than current period.
            elif (self.N - self.t) <= 0:
              raise ValueError("You must provide a period interval at least >= 0")
            return self.euro_price()

        elif self.exercise_type == 'amer':
            if (self.N - self.t) == 0:

                if self.option_type == "call":
                    # If the option is a call type, we will return a max of the spot price less the strike price. No need to calculate further as there is no step period.
                    return max(self.S - self.K,0)

                elif self.option_type == "put":
                    # If the option type is a put type, returns a max of the strike price less the spot price. No need to calculate further as there is no step period.
                    return max (self.K - self.S,0)

                else:
                    # This ensures the user will input one option type. Since this is a required input type, we opted to raise an error rather than set a default value.
                    raise ValueError("You must provide an option type of either call or put. try again.")
            # Handles possibility of N being less than current period.
            elif (self.N - self.t) <= 0:
              raise ValueError("You must provide a period interval at least >= 0")

              return self.binomial_american()
            return self.binomial_american()
        else:
            raise ValueError("Exercise type must be 'euro' or 'amer'.")





