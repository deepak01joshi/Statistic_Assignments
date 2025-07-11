# Statistics Advance Part 1 - Theory Answers in Simple Language

# 1. What is a random variable?
"""
A random variable is a value that comes from a random experiment. It can change each time you repeat the experiment.
Example: Rolling a die – the result (1 to 6) is a random variable.
"""

# 2. Types of random variables
"""
- Discrete: Has specific values (e.g., 0, 1, 2)
- Continuous: Can take any value in a range (e.g., 1.1, 2.35)
"""

# 3. Discrete vs. Continuous distributions
"""
- Discrete: Probabilities assigned to exact values
- Continuous: Probabilities over intervals, not exact points
"""

# 4. What are probability distribution functions (PDF)?
"""
A PDF tells how likely a random variable is to take on a certain value.
"""

# 5. CDF vs. PDF
"""
- PDF: Probability at a point
- CDF: Total probability up to a point
"""

# 6. Discrete uniform distribution
"""
All outcomes have equal probability. Example: Rolling a fair die.
"""

# 7. Bernoulli distribution
"""
Two outcomes: success (1) or failure (0). Key property: Single trial.
"""

# 8. Binomial distribution
"""
Used when we repeat Bernoulli trials multiple times (e.g., flipping a coin 10 times).
"""

# 9. Poisson distribution
"""
Counts how many times an event happens in a fixed interval. Used in call centers, traffic flow, etc.
"""

# 10. Continuous uniform distribution
"""
All values in a range are equally likely (e.g., picking a number between 0 and 1).
"""

# 11. Characteristics of a normal distribution
"""
- Bell-shaped
- Symmetrical
- Mean = Median = Mode
"""

# 12. Standard normal distribution
"""
Normal distribution with mean = 0 and std dev = 1. Used for Z-scores.
"""

# 13. Central Limit Theorem (CLT)
"""
When you take many samples from any distribution, the sample means form a normal distribution.
"""

# 14. CLT and normal distribution
"""
CLT explains why normal distribution appears in many real-world situations.
"""

# 15. Application of Z-statistics
"""
Used to test if a sample is significantly different from a population.
"""

# 16. Z-score meaning
"""
Z = (X - Mean) / Std Dev
It tells how far a value is from the mean.
"""

# 17. Point vs. Interval estimates
"""
- Point: Single value estimate
- Interval: Range of possible values
"""

# 18. Significance of confidence intervals
"""
They show how certain we are about the estimate (e.g., 95% sure).
"""

# 19. Z-score and confidence interval
"""
Z-scores help calculate confidence intervals.
"""

# 20. Comparing different distributions using Z-scores
"""
Z-scores standardize values, so we can compare different scales.
"""

# 21. Assumptions of CLT
"""
- Samples are independent
- Sample size is large enough (usually > 30)
"""

# 22. Expected value
"""
The average outcome if we repeat the experiment many times.
"""

# 23. Expected value and distribution
"""
Expected value is the center (mean) of a distribution.
"""

# ========================================
# Python Programs for Probability and Distributions
# ========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, norm, uniform, zscore
import seaborn as sns

# Generate a random variable and display its value
rand_val = np.random.randint(1, 100)
print("Random Variable:", rand_val)

# Generate a discrete uniform distribution and plot PMF
values = np.arange(1, 7)
pmf = [1/6] * 6
plt.bar(values, pmf)
plt.title("Discrete Uniform Distribution PMF")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()

# Bernoulli PDF
def bernoulli_pdf(p):
    x = [0, 1]
    return bernoulli.pmf(x, p)

# Simulate a binomial distribution and plot histogram
binom_data = binom.rvs(n=10, p=0.5, size=1000)
plt.hist(binom_data, bins=10, edgecolor='black')
plt.title("Binomial Distribution Histogram")
plt.show()

# Poisson distribution and visualization
poisson_data = poisson.rvs(mu=3, size=1000)
sns.histplot(poisson_data, bins=15)
plt.title("Poisson Distribution")
plt.show()

# CDF of discrete uniform
cdf = np.cumsum(pmf)
plt.step(values, cdf)
plt.title("CDF of Discrete Uniform Distribution")
plt.show()

# Continuous uniform distribution using NumPy and visualize
cont_uniform = np.random.uniform(0, 1, 1000)
plt.hist(cont_uniform, bins=20)
plt.title("Continuous Uniform Distribution")
plt.show()

# Simulate normal distribution and plot histogram
data_norm = np.random.normal(0, 1, 1000)
plt.hist(data_norm, bins=30)
plt.title("Normal Distribution Histogram")
plt.show()

# Calculate and plot Z-scores
def plot_z_scores(data):
    zs = zscore(data)
    plt.hist(zs, bins=30)
    plt.title("Z-Score Distribution")
    plt.show()
    return zs

z_scores = plot_z_scores(data_norm)

# Implement CLT using non-normal distribution (Exponential)
sample_means = [np.mean(np.random.exponential(scale=2.0, size=30)) for _ in range(1000)]
plt.hist(sample_means, bins=30)
plt.title("CLT with Exponential Distribution")
plt.show()

# Continued Practical Questions

# Simulate multiple samples from a normal distribution and verify CLT
samples = [np.mean(np.random.normal(loc=50, scale=10, size=30)) for _ in range(1000)]
plt.hist(samples, bins=30, edgecolor='black')
plt.title("CLT Verification with Normal Distribution")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()

# Function to calculate and plot standard normal distribution
def plot_standard_normal():
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    plt.plot(x, y)
    plt.title("Standard Normal Distribution")
    plt.xlabel("Z")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.show()

plot_standard_normal()

# Generate binomial probabilities
def binomial_probabilities(n=10, p=0.5):
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    plt.bar(x, pmf)
    plt.title("Binomial Distribution PMF")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.show()
    return list(zip(x, pmf))

print("Binomial PMF:", binomial_probabilities())

# Calculate Z-score for a given point and compare to standard normal
def calculate_z_score(value, mean, std_dev):
    z = (value - mean) / std_dev
    print(f"Z-score of {value} with mean {mean} and std dev {std_dev}: {z:.2f}")
    return z

calculate_z_score(75, 70, 10)

# Hypothesis testing using Z-statistics
sample_mean = 72
population_mean = 70
sample_std = 10
n = 50
z_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
print("Z-statistic:", z_stat)

# Confidence interval for dataset
z_critical = norm.ppf(0.975)  # 95% confidence
margin_error = z_critical * (sample_std / np.sqrt(n))
ci = (sample_mean - margin_error, sample_mean + margin_error)
print("95% Confidence Interval:", ci)

# Generate normal data and interpret CI for its mean
data = np.random.normal(50, 10, 100)
mean_data = np.mean(data)
std_err = np.std(data, ddof=1) / np.sqrt(len(data))
ci_data = (mean_data - z_critical * std_err, mean_data + z_critical * std_err)
print("Mean CI for normal data:", ci_data)

# PDF of normal distribution
x = np.linspace(-4, 4, 100)
pdf = norm.pdf(x)
plt.plot(x, pdf)
plt.title("PDF of Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# CDF of Poisson distribution
x = np.arange(0, 20)
mu = 4
cdf_poisson = poisson.cdf(x, mu)
plt.step(x, cdf_poisson, where='mid')
plt.title("CDF of Poisson Distribution")
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.show()

# Simulate continuous uniform variable and expected value
uniform_data = np.random.uniform(10, 20, 1000)
print("Expected Value of Uniform Data:", np.mean(uniform_data))

# Compare standard deviations of two datasets
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(0, 2, 1000)
plt.hist(data1, bins=30, alpha=0.5, label="std=1")
plt.hist(data2, bins=30, alpha=0.5, label="std=2")
plt.title("Comparison of Standard Deviations")
plt.legend()
plt.show()

# Calculate range and IQR
range_ = np.ptp(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)
print("Range:", range_)
print("IQR:", iqr)

# Z-score normalization and visualization
normalized_data = zscore(data)
plt.hist(normalized_data, bins=30)
plt.title("Z-score Normalized Data")
plt.grid(True)
plt.show()

# Skewness and kurtosis
from scipy.stats import skew, kurtosis
print("Skewness:", skew(data))
print("Kurtosis:", kurtosis(data))

