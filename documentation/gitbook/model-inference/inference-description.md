---
description: Methods for fitting model to data
---

# Inference Description

_flepiMoP_ can be used to conduct forward simulations of a model with user-defined parameter values, or, it can be used to iteratively run a model with some unknown parameters, compare the model output to ground truth data, and find parameter values that optimize the fit of the model to data (i.e., conduct model "inference"). We have developed a custom model inference method that is based on standard Markov Chain Monte Carlo (MCMC)-based approaches to Bayesian inference for dynamic models, but is adapted to deal with some of the particular challenges of large-scale epidemic models, including i) long times and high computational resources required to simulate single model runs, ii) multiple subpopulations with location-specific parameters but inter-location transmission, iii) a high-dimensional parameter space, iv) the need to produce real-time epidemic projections, and v) the availability of parallel computing resources.

## Notation

* $$\Theta$$ – A set of unknown **model parameters** to be estimated by fitting the model output to data. For a model with $$i$$ subpopulations each with their own parameters, this set includes all location-specific parameters $$\Theta_i$$.
* $$Z(\Theta)$$ – The timeseries output of one or more of the state **variables of the model** under parameters $$\Theta.$$ For simplicity, we will often just use the notation $$Z$$. The value at a timepoint $$t$$is $$Z_t$$. For a model with $$i$$ subpopulations for which there are $$j$$ different state variables, this becomes $$Z_{i,j,t}(\Theta)$$. (Note that for the general case when the dynamics in one location can effect the dynamics in another, the model state in one location depends on the full set of parameters, not just the location-specific parameters.)
* $$D_t$$ – The timeseries for the **observed data** (also referred to as "ground truth") that the model attempts to recreate. For a model with $$i$$ subpopulations each with their own observed data for variable $$j$$, this becomes $$D_{i,j,t}$$.
* $$\mathcal{L}(D|\Theta)$$ – The **likelihood** of the observed data $$D$$being produced by the model for an input parameter set $$\Theta$$. This is a probability density function over all possible values of the data being produced by the model, conditional on a fixed model parameter value ;
* $$p(\Theta)$$ – The **prior probability** distribution, which in Bayesian inference encodes beliefs about the possible values of the unknown parameter $$\Theta$$ before any observed data is formally compared to the model.
* $$P(\Theta|D)$$ – The **posterior probability** distribution, which in Bayesian inference describes the updated probability of the parameters $$\Theta$$ conditional on the observed data $$D$$.
* $$g(\Theta^*|\Theta)$$ – The **proposal density**, used in Metropolis-Hastings algorithms for Markov Chain Monte Carlo (MCMC) techniques for sampling the posterior distribution, describes the probability of proposing a new parameter set $$\Theta^*$$ from a current accepted parameter set $$\Theta$$.

## Background

This section can be skipped by those familiar with Markov Chain Monte Carlo approaches to Bayesian inference.

### Bayesian inference

Our model fitting framework is based on the principles of Bayesian inference. Instead of estimating a single "best-fit" value of the unknown model parameters, our goal is to evaluate the consistency of every possible parameter value with the observed data, or in other words, to construct a distribution that describes the probability that a parameter has a certain value given the observations. This output is referred to as the _posterior probability._ This framework assumes that the model structure accurately describes the underlying generative process which created the data, but that the underlying parameters are unknown and that there can be some error in the observation of the data.

Bayes' Rule states that the posterior probability $$P(\Theta|D)$$ of a set of model parameters $$\Theta$$ given observed data $$D$$ can be expressed as a function of the likelihood of observing the data under the model with those parameters ($$\mathcal{L}(D|\Theta)$$) and the prior probability ascribed to those parameters before any data was observed ($$p(\Theta)$$)

$$
P(\Theta|D) = \frac{\mathcal{L}(D|\Theta)p(\Theta)}{P(D)}
$$

where the denominator  $$P(D) = \int_\Theta \mathcal{L}(D|\Theta)p(\Theta) d\Theta$$ is a constant factor – independent of $$\Theta$$ – that only serves to normalize the posterior and thus can be ignored ;

The likelihood function can be defined for a model/data combination based on an understanding of both a) the distribution of model outcomes for a given set of input parameters (if output is stochastic), and b) the nature of the measurement error in observing the data (if relevant) ;

For complex models with many parameters like those used to simulate epidemic spread, it is generally impossible to construct the full posterior distribution either analytically or numerically. Instead, we rely on a class of methods called "Markov Chain Monte Carlo" (MCMC) that allows us to draw a random sample of parameters from the posterior distribution. Ideally, the statistics of the parameters drawn from this sample should be an unbiased estimate of those from the complete posterior.

### Markov Chain Monte Carlo methods

In many Bayesian inference problems that arise in scientific model fitting, it is impossible to directly evaluate the full posterior distribution, since there are many parameters to be inferred (high dimensionality) and it is computationally costly to evaluate the model at any individual parameter set. Instead, it is common to employ Markov Chain Monte Carlo (MCMC) methods, which provide a way to iteratively construct a sequence of values that when taken together represent a sample from a desired probability distribution. In the limit of infinitely long sequences ("chains") of values, these methods are mathematically proven to converge to an unbiased sample from the distribution. There are many different MCMC algorithms, but each of them relies on some type of rule for generating a new "sampled" parameter set from an existing one. Our parameter inference method is based on the popular Metropolis-Hastings algorithm. Briefly, at every step of this iterative algorithm, a new set of parameters is jointly proposed, the model is evaluated at that proposed set, the value of the posterior (e.g., likelihood and prior) is evaluated at the proposed set, and if the posterior is improved compared to the previous step, the proposed parameters are "accepted" and become the next entry in the sequence, whereas if the value of the posterior is decreased, the proposed parameters are only accepted with some probability and otherwise rejected (in which case the next entry in the sequences becomes a repeat of the previous parameter set).

The full algorithm for Metropolis-Hastings Markov Chain Monte Carlo is:

* Generate initial set of parameters $$\Theta_0$$
* Evaluate the likelihood ($$\mathcal{L}(D|\Theta)$$) and prior ($$p(\Theta)$$) at this parameter set
* `For` $$k = 1 \cdots K$$ where $$K$$ is the length of the MCMC chain, add to the sequence of parameter values :
  * Generate a proposed set of parameters $$\Theta^*$$ based on an arbitrary proposal distribution $$g(\Theta^*|\Theta_{k-1})$$
  * Evaluate the likelihood and prior at the proposed parameter set
  * Generate a uniform random number $$u \sim \mathcal{U}[0,1]$$
  * Calculate the acceptance ratio $$\alpha=\frac{\mathcal{L}(D|\Theta^*) p(\Theta^*) g(\Theta_{k-1}|\Theta^*)}{\mathcal{L}(D|\Theta_{k-1})) p(\Theta_{k-1}) g(\Theta^*|\Theta_{k-1})}$$
  * `If` $$\alpha> u$$, ACCEPT the proposed parameters to the parameter chain. Set $$\Theta_k=\Theta^*$$ ;
  * `Else,` REJECT the proposed parameters for the chimeric parameter chain. Set $$\Theta_k = \Theta_{k-1}$$

## Inference algorithm

### Likelihood

In our algorithm, model fitting involves comparing timeseries of variables produced by the model (either transmission model state variables or observable outcomes constructed from those variables) to timeseries of observed "ground truth" data with the same time points. For timeseries data that arises from a deterministic, dynamic model, then the overall likelihood can be calculated as the product of the likelihood of the model output at each timepoint (since we assume the data at each timepoint was measured independently). If there are multiple observed datastreams corresponding to multiple model outputs (e.g., cases and deaths) ;

For each subpopulation in the model, the likelihood of observing the "ground truth" data given the model parameters $$\Theta$$ is

$$
\mathcal{L}_i(D_i|\Theta) = \mathcal{L}(D_i|Z_i(\Theta)) = \prod_j \prod_t p^{\text{obs}}_j(D_{i,j,t}|Z_{i,j,t}(\Theta))
$$

where $$p^{\text{obs}}(D|Z)$$ describes the process by which the data is assumed to be observed/measured from the underlying true vales. For example, observations may be assumed to be normally distributed around the truth with a known variance, or, count data may be assumed to be generated by a Poisson process ;

And the overall likelihood taking into account all subpopulations, is the product of the individual likelihoods

$$
\mathcal{L}(D|\Theta) =\mathcal{L}(D|Z(\Theta)) =  \prod_i \mathcal{L}_i(D_i|Z_i(\Theta))
$$

Note that the likelihood for each subpopulation depends not only on the parameter values $$\Theta_i$$ that act within that subpopulation, but on the entire parameter set $$\Theta$$, since in general the infection dynamics in one subpopulation are also affected by those in each other region. Also note that we assume that the parameters $$\Theta$$ only impact the likelihood through the single model output timeseries $$Z_t$$. While this is exactly true for a deterministic model, we make the simplifying assumption that it is also true for stochastic models, instead of attempting to calculate the full distribution of possible trajectories for a given parameter set and include that in the likelihood as well.

### Fitting algorithm

The method we use for estimating model parameters is based on the Metropolis-Hastings algorithm, which is a class of Markov Chain Monte Carlo (MCMC) methods for obtaining samples from a posterior probability distribution. We developed a custom version of this algorithm to deal with some of the particular mathematical properties and computational challenges of fitting large disease transmission models ;

There are to major unique features of our adapted algorithm:

* **Parallelization** – Generally MCMC methods starting from a single initial parameter set and generating an extremely long sequence of parameter samples such that the Markov process is acceptably close to a stationary state where it represents an unbiased sample from the posterior. Instead, we s**imulate multiple shorter chains in parallel**, starting from different initial conditions, and pool the results. Due to the computational time required to simulate the epidemic model, and the timescale on which forecasts of epidemic trajectories are often needed (\~weeks), it is not possible to sequentially simulate the model millions of times. However, modern supercomputers allow massively parallel computation. The hope of this algorithm is that the parallel chains sample different subspaces of the posterior distribution, and together represent a reasonable sample from the full posterior. To maximize the chance of at least local stationarity of these subsamples, we pool only the final values of each of the parallel chains.
* **Multi-level** – Our pipeline, and the fitting algorithm in particular, were designed to be able to simulate disease dynamics in a collection of linked subpopulations. This population structure creates challenges for model fitting. We want the model to be able to recreate the dynamics in each subpopulation, not just the overall summed dynamics. Each subpopulation has unique parameters, but due to the coupling between them (), the model outcomes in one subpopulation also depend on the parameter values in other subpopulations.  For some subpopulations (), this coupling may effectively be weak and have little impact on dynamics, but for others (), spillover from another closely connected subpopulation may be the primary driver of the local dynamics. Thus, the model cannot be separately fit to each subpopulation, but must consider the combined likelihood. However, such an algorithm may be very slow to find parameters that optimize fits in all locations simultaneously, and may be predominantly drawn to fitting to the largest/most connected subpopulations. The avoid these issues, we **simultaneously generate two communicating parameter chains**: a "chimeric" chain that allows the parameters for each subpopulation to evolve quasi-independently based on local fit quality, and a "global" chain that evolves only based on the overall fit quality (for all subpopulations combined).

Note that while the traditional Metropolis-Hastings algorithm for MCMC will provably converge to a stationary distribution where the sequence of parameters represents a sample from the posterior distribution, no such claim has been mathematically proven for our method.

* `For` $$m=1  \dots M$$, where $$M$$is the number of parallel MCMC chains (also known as _slots_)
  * Generate initial state
    * Generate an initial set of parameters $$\Theta_{m,0}$$, and copy this to both the global ($$\Theta^G_{m,0}$$) and chimeric ($$\Theta^C_{m,0}$$) parameter chain (sequence ;
    * Generate an initial epidemic trajectory $$Z(\Theta_{m,0})$$
    * Calculate and record the initial likelihood for each subpopulation, $$\mathcal{L_i}(D_i|Z_i(\Theta_{m,0}))$ ;
  * `For` $$k= 1 ... K$$ where $$K$$ is the length of the MCMC chain, add to the sequence of parameter values :
    * Generate a proposed set of parameters $$\Theta^*$$from the current chimeric parameters using the proposal distribution $$g(\Theta^*|\Theta^C_{m,k-1})$ ;
    * Generate an epidemic trajectory with these proposed parameters, $$Z(\Theta^*)$$
    * Calculate the likelihood of the data given the proposed parameters for each subpopulation, $$\mathcal{L}_i(D_i|Z_i(\Theta^*))$$
    * Calculate the overall likelihood with the proposed parameters, $$\mathcal{L}(D|Z(\Theta^*))$$
    * Make "global" decision about proposed parameters
      * Generate a uniform random number $$u^G \sim \mathcal{U}[0,1]$$
      * Calculate the overall likelihood with the current global parameters, $$\mathcal{L}(D|Z(\Theta^G_{m,k-1}))$$
      * Calculate the acceptance ratio $$\alpha^G=\min \left(1, \frac{\mathcal{L}(D|Z(\Theta^*)) p(\Theta^*) }{\mathcal{L}(D|Z(\Theta^G_{m,k-1})) p(\Theta^G_{m,k-1}) } \right)$$​
      * `If` $$\alpha^G > u^G$$: ACCEPT the proposed parameters to the global and chimeric parameter chains
        * Set $$\Theta^G_{m,k} =$$$$\Theta^*$$
        * Set $$\Theta_{m,k}^C=\Theta^*$$
        * Update the recorded subpopulation-specific likelihood values (chimeric and global) with the likelihoods calculated using the proposed parameter ;
      * `Else`: REJECT the proposed parameters for the global chain and make subpopulation-specific decisions for the chimeric chain
        * Set $$\Theta^G_{m,k} = \Theta^G_{m,k-1}$$
        * Make "chimeric" decision:
          * `For` $$i = 1 \dots N$$
            * Generate a uniform random number $$u_i^C \sim \mathcal{U}[0,1]$$
            * Calculate the acceptance ratio $$\alpha_i^C=\frac{\mathcal{L}_i(D_i|Z_i(\Theta^*)) p(\Theta^*) }{\mathcal{L}i(D_i|Z_i(\Theta^C_{m,k-1})) p(\Theta_{m,k-1}) }$$
            * `If` $$\alpha_i^C > u_i^C$$: ACCEPT the proposed parameters to the chimeric parameter chain for this location
              * Set $$\Theta_{m,k,i}^C = \Theta^*_{i}$$
              * Update the recorded chimeric likelihood value for subpopulation $$i$$ to that calculated with the proposed parameter​
            * `Else`: REJECT the proposed parameters for the chimeric parameter chain for this location
              * Set $$\Theta_{m,k,i}^C=\Theta_{m,k-1,i}$$​
            * `End if ;
          * `End for` $$N$$subpopulations
        * End making chimeric decisions
      * `End if`
    * End making global decision
  * `End for` $$K$$ iterations of each MCMC chain
* `End for` $$M$$ parallel MCMC chains
* Collect the final global parameter values for each parallel chain $$\theta_m = \{\Theta^G_{m,K}\}_m$$

We consider the sequence $$\theta_m$$ to represent a sample from the posterior probability distribution, and use it to calculate statistics about the inferred parameter values and the epidemic trajectories resulting from them (e.g., mean, median, 95% credible intervals).

<figure><img src="../.gitbook/assets/FlepiMop Inference.png" alt=""><figcaption><p>Diagram of the custom multi-level MCMC method used for parameter inference in <em>flepiMoP</em>. Each square represents a single subpopulation which has a set of associated parameter values. Diagram is for a single MCMC chain; multiple parallel chains are typically run and combined to form a posterior distribution of parameter values.</p></figcaption></figure>

There are a few important notes/limitations about our method currently:

* All parameters to be fit must be location-specific. There is currently no way to fit a parameter that has the identical value across all locations
* The pipeline currently does not allow for fitting of the basic parameters of the compartmental epidemic model. Instead, these must be fixed, and the value of location-specific "interventions"  acting to increase/reduce these parameters can be fit. All parameters related to the observational/outcomes model can be fit, as well as "interventions" acting to increase or reduce them ;
* At no point is the parameter fitting optimizing the fit of the summed total population data to total population model predictions. The "overall" likelihood function used to make "global" parameter acceptance decisions is the product of the individual subpopulations likelihoods (which are based on comparing location-specific data to location-specific model output), which is not equivalent to likelihood for the total population. For example, if overestimates of the model in some subpopulations were exactly balanced by underestimates in others, the total population estimate could be very accurate and the total population likelihood high, but the overall likelihood we use here would still be low.
* There is no model simulation run or record that corresponds to the combined parameters recorded in the chimeric parameter chain ($$\Theta^C_{m}$$). For entry $$m$$ in the chain, some of these parameter values were recently accepted from the last proposal and were used in the simulation produced by that proposal, while for other subpopulations, the most recent proposed parameters were rejected so $$\Theta^C_{m}$$ contains parameters accepted – and part of the simulations produced – in a previous iteration.
* It is currently not possible to infer parameters of the measurement process encoded in the likelihood function. For example, if the likelihood is chosen to be a normal distribution, which implies an assumption that the observed data is generated from the underlying truth according to a normal distribution with mean zero, then the standard deviation must be specified, and cannot be inferred along with the other model parameters ;
* There is an option to use a slightly different version of our algorithm, in which globally accepted parameter values are not pushed back into the chimeric likelihood, but the chimeric likelihood is instead allowed to continue to evolve independently. In this variation, the chimeric acceptance decision is always made, not only if a global rejection happens ;
* The proposal distribution $$g(\Theta^*|\Theta)$$ for generating new parameter sets is currently constrained to be a joint distribution in which the the value of each new proposed parameter is chosen independently of any other parameters.
* While in general in Metropolis-Hasting algorithms the formula for the the acceptance ratio includes the proposal distribution $$g(\Theta^*|\Theta)$$, those terms cancel out if the proposal distribution is symmetrical. Our algorithm assumes such symmetry and thus does not include $$g$$ in the formula, so the user must be careful to only select symmetric distributions.

### Hierarchical parameters

The baseline likelihood function used in the fitting algorithm described above allows for parameter values to differ arbitrarily between different subpopulations. However, it may be desired to instead impose constraints on the best-fit parameters, such that subpopulations that are similar in some way, or belong to some pre-defined group, have parameters that are close to one another. Formally, this is typically done with group-level or hierarchical models that fit meta-parameters from which individual subpopulation parameters are assumed to draw. Here, we instead impose this group-level structure by adding an additional term to the likelihood that describes the probability that the set of parameters proposed for a group of subpopulations comes from a normal distribution. This term of the likelihood will be larger when the variance of this parameter set is smaller. Formally

$$
\mathcal{L}(D|\Theta)  \rightarrow  \prod_i \mathcal{L}_i(D_i|Z_i(\Theta))  \cdot \prod_g \prod_{i \in g} \prod_l \phi(\Theta_{l,i}; \mu_{l,g}, \sigma_{l,g})
$$

where $$g$$ is a group of subpopulations, $$l$$ is one of the parameters in the set $$\Theta$$, $$\varphi(x;\mu,\sigma)$$is the probability density function of the normal distribution, $$\mu_{l,g}$$ and $$\sigma_{l,g}$$ are the mean and standard deviation of all values of the parameter $$\Theta_l$$ in the group $$g$$. There is also the option to use a logit-normal distribution instead of a standard normal, which may be more appropriate if the parameter is a proportion bounded in \[0,1].

###
