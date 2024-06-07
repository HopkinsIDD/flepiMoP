---
description: >-
  This is just a place to play around with different inference algorithms.
  Gitbook markdown is very application-specific so can't copy this algorithm
  text into other apps to play around with!
---

# Inference scratch

### Current inference algorithm

* `For` $$m=1  \dots M$$, where $$M$$is the number of parallel MCMC chains (also known as _slots_)
  * Generate initial state
    * Generate an initial set of parameters $$\Theta_{m,0}$$, and copy this to both the global ($$\Theta^G_{m,0}$$) and chimeric ($$\Theta^C_{m,0}$$) parameter chain (sequence)&#x20;
    * Generate an initial epidemic trajectory $$Z(\Theta_{m,0})$$
    * Calculate and record the initial likelihood for each subpopulation, $$\mathcal{L_i}(D_i|Z_i(\Theta_{m,0}))$$&#x20;
  * `For` $$k= 1 ... K$$ where $$K$$ is the length of the MCMC chain, add to the sequence of parameter values :
    * Generate a proposed set of parameters $$\Theta^*$$from the current chimeric parameters using the proposal distribution $$g(\Theta^*|\Theta^C_{m,k-1})$$&#x20;
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
        * Update the recorded subpopulation-specific likelihood values (chimeric and global) with the likelihoods calculated using the proposed parameters&#x20;
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
            * `End if`&#x20;
          * `End for` $$N$$subpopulations
        * End making chimeric decisions
      * `End if`
    * End making global decision
  * `End for` $$K$$ iterations of each MCMC chain
* `End for` $$M$$ parallel MCMC chains
* Collect the final global parameter values for each parallel chain $$\theta_m = \{\Theta^G_{m,K}\}_m$$

## Making chimeric decision first
