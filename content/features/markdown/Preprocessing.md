# Recommended Preprocessing

There are currently no agreed upon conventions for preprocessing naturalistic neuroimaging data. Here different speakers share their current recommendations.


# Luke Chang
In our work, we perform standard realignment and spatial normalization. We also perform basic denosing by removing global and multivariate spikes, average CSF activity, linear/quadratic trends, and 24 motion covariates (e.g., 6 centered realignment, their squares, derivative, and squared derivatives). We are very cautious about performing high-pass filtering as many of the effects we are interested in occur in slower frequencies. We find that including average activity from a CSF mask helps a lot in reducing different types of physiological and motion related artifacts. We typically apply spatial smoothing, but depending on the question we don't always perform this step. For spatial feature selection, we rarely use searchlights and instead use parcellations. This allows us to quickly prototype analysis ideas using smaller numbers of parcels (e.g., n=50) and then increase the number if we want greater spatial specificity. We usually use a [parcellation](https://neurovault.org/collections/2099/) that we developed based on meta-analytic coactivation using the neurosynth database.
