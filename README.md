
# Ship Accidents: Regression Modeling and Assumption Testing in R

**Question:** What drives the number of damage incidents on ships, and does a normal linear regression model meet its assumptions on this data?

**Data:** `ShipAccidents.csv` contains 40 ship groups. Each group is defined by a ship type (`construction1`–`construction3`, with one baseline type) and an operating period (`operational`). The response is `accidents`, the incident count. The predictors are `exposure` and the construction type, and `service_months` is aggregate months in service. Six groups with no service were dropped, leaving **n = 34**.

**Methods:** multiple linear regression, F and t tests, residual diagnostics, Shapiro-Wilk normality test, the Cook-Weisberg / Breusch-Pagan score test for non-constant variance (`car::ncvTest`), a variance-stabilizing square-root transformation, and weighted least squares (WLS).

## Key findings

| Model | R² | Overall F-test p | Normality (Shapiro-Wilk p) | Constant variance (ncvTest) |
|---|---|---|---|---|
| OLS on `accidents` | 0.73 | 6.0e-08 | 0.073 ✓ | χ² = 7.39, **p = 0.0065 ✗** |
| OLS on `sqrt(accidents)` | 0.83 | 6.2e-11 | 0.554 ✓ | χ² = 0.38, p = 0.537 ✓ |
| WLS on `accidents`, weights = √service_months | 0.85 | — | 0.080 ✓ | χ² = 1.33, p = 0.250 ✓ |

1. **Exposure is the dominant driver of accidents** (slope p = 1.3e-09). Ships with more exposure have significantly more incidents.
2. **The untransformed model breaks the constant-variance assumption.** The residual spread grows with the fitted values, so its standard errors and p-values can't be fully trusted.
3. **The square-root transformation fixed the problem.** It passed both the variance and normality tests, its residual plot looks like random noise, and fit improved from R² 0.73 to 0.83. **This is the recommended model.**
4. **WLS passed the formal test, but I don't recommend it.** WLS weights are meant to be proportional to *1 / variance*. Weighting by √service_months gives *more* weight to ships with more service, which implies those ships have *less* variable accident counts. That is the opposite of what we would expect for count data. When I used the inverse weights (1/√service_months), the test failed (p = 0.0015). In addition, the residual plot still shows a pattern.
5. **Next step:** Accidents are counts, so a **Poisson or negative binomial regression with log(service_months) as an offset** fits the data more naturally than a normal linear model. A Poisson fit shows overdispersion (deviance/df ≈ 2.1), which points to negative binomial.

---

## Full analysis


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown
## Load data into R

## Read the data into R in both txt and csv format

```{r text}
ShipData = read.table('ShipAccidents.txt',header=TRUE)
```
```{r csv}
ShipData = read.csv('ShipAccidents.csv',header=TRUE)
```

## Confirm missing values have been accounted for correctly

```{r confirm}
ShipData = na.omit(ShipData)
attach(ShipData)
ShipData
```

<img width="839" alt="DF" src="https://github.com/user-attachments/assets/ed496a8c-c36e-431f-94b3-c5463e509be2">


## Descriptive statistics to confirm the data has been read in correctly

```{r descriptive}
hist(accidents)
boxplot(accidents)
mean(accidents)
var(accidents)
summary(accidents)
table(accidents)
```

<img width="561" alt="Hist" src="https://github.com/user-attachments/assets/31d6088b-202c-4cdb-8738-2fd8f43ae5eb">

<img width="693" alt="Box plot" src="https://github.com/user-attachments/assets/372474cb-d7f8-45d7-ad9b-ca0694013861">

## Present an appropriate normal linear regression model

```{r regression}
NormalModel = lm(accidents~exposure+construction1+construction2+construction3)
summary(NormalModel)
```
<img width="934" alt="Lm res" src="https://github.com/user-attachments/assets/2ce3f08d-37b2-46c1-9e3e-6e1864a827f8">

accidents$_i$ = -50.7824 + 7.9985 exposure + 8.0606 construction$_1$ + 7.2704 construction$_2$ + 2.0311 construction$_3$ + $\epsilon_i$

## List the assumptions of the model (Linear Regression)

- The normal linear regression model has the four standard assumptions:

1. The linear relationship is appropriate.

2. The responses are independent.

3. The responses are normally distributed.

4. The responses have equal variation.

## Discussion of all the assumptions of the model, including tests of the assumptions

- The assumptions have the following interpretations:

1. The linear relationship is appropriate: This states that the linear function relating the parameters (the $\beta's$) to the mean of the response is reasonable. This is often evaluated by considering residual plots: a pattern in the plot may suggest that the proposed relationship is not appropriate.

2. The responses are independent: This states that the responses are distributed independently of each other. This is typically evaluated by considering the sampling technique. Tests are available in special cases.

3. The responses are normally distributed: This states that the shape of the distribution of all responses is normal. This is typically tested by classical normality tests applied to the residuals, along with normal probability plots.

4. The responses have equal variation: This states the the responses come from distributions with the same variation. This is typically tested in regression using White's Test or the Breusch-Pagan test, both of which consider the effects of individual predictors on response variation within the model. Residual plots are
also considered.

## Tests of the assumptions

- a. Linearity of regression function 

```{r linearity non formal}
library(ggplot2)
qplot(NormalModel$fitted.values,NormalModel$residuals)
``` 

<img width="727" alt="ggplot1" src="https://github.com/user-attachments/assets/a58237dd-09cc-4f06-8c5f-a92a4a968744">

Conclusion: The residuals do not scatter randomly around zero. They fan out as the fitted values increase, so the plot alone does not confirm that the linear form is appropriate. It also points to the non-constant variance tested below.

```{r linearity formal}
NormalModel = lm(accidents~exposure+construction1+construction2+construction3)
summary(NormalModel)
```

Hypothesis (overall F-test):

 $H_0: \beta_1 = \beta_2 = \beta_3 = \beta_4 = 0$ (no predictor is related to accidents)
 
 $H_A$: at least one $\beta_j ≠ 0$
 
Conclusion: The overall F-test p-value is 5.957e-08, which is below $\alpha$ = 0.05. We reject $H_0$: the predictors together are significantly related to accidents. The slope for `exposure` is significant on its own (t-test p = 1.3e-09). Significance alone does not prove that the linear *form* is correct, so the residual plots above are still needed to check it.

- b. Normality of variables

```{r normality}
shapiro.test(NormalModel$residuals)
```

<img width="744" alt="Shapiro" src="https://github.com/user-attachments/assets/de4bc5f9-bd84-4bb1-93ca-9d6e67bf0bc1">


 Hypothesis:
 
 $H_o$: Normal
 
 $H_a$: Not normal

Conclusion: The p-value is 0.0729, which is greater than $\alpha$ = 0.05. We fail to reject $H_0$, so there is no significant evidence that the residuals are non-normal.

- c. Non-constant Variance

```{r nonconstant}
library(car)
ncvTest(NormalModel)
```

<img width="468" alt="Non CNS" src="https://github.com/user-attachments/assets/27c880f6-78b2-4b8c-ba26-78760013db49">


 Hypothesis:
 
 $H_o$: constant variance
 
 $H_a$: Non constant variance

Conclusion: The p-value is 0.0065 (χ² = 7.39), which is less than $\alpha$ = 0.05. We reject $H_0$ and conclude that the error variance is **not constant** (heteroscedasticity).
 
- d. Independence of error terms 

conclusion: Since their is no time order given to the data, we assume independence of error terms. 

## Describe the meaning of Homogeneity of Variance
Homogeneity of variance: Describes a situation in which the error term is 
the same across all values of the independent variables which are exposure, construction 1,construction 2 and construction 3.

consequences of ignoring violations of this assumption

We cannot apply the formula of the variance of the coefficient to conduct tests of significance and construct confidence intervals.

If μ (error term) is heteroscedastic, the OLS estimates will be inefficient in small samples

The coefficient estimates would still be statistically unbiased even if the μ‘s are heteroscedastic

The prediction of Y for a given value of X based on the estimates from the original
data, would have a high variance.

OR

This assumption means that each observed response comes from a distribution with the same level of variation, regardless of the specific response. In other words, while each response has some inherent variability, this variability remains constant across all observations. Although it doesn’t explicitly state that the variances across groups are equal, this is a natural consequence of the assumption. In this context, it implies that the "number of accidents" is drawn from a distribution with consistent, unchanging variation.

When applying the ncvTest to the proposed normal model, the results provide evidence to reject the assumption of constant variance (χ² = 7.39, p = 0.0065). Furthermore, the residuals versus predicted values plot suggests increasing variance as predicted values rise. This plot indicates additional problems, such as a potential misfit in the relationship or missing predictors. Overall, the test shows signs of non-constant variation. If this violation is ignored, the actual distribution of the F-statistic becomes uncertain, making hypothesis tests potentially misleading.

## Process of Using Transformation 

- Identify the type of transformation that is most effective  

- After transformation, rerun the ANOVA on the transformed data  

 - Recheck the transformed data against the assumptions for the ANOVA  
 
 - Lastly, if the transformation is significant, then keep it. If not, disregard it  
 
## Purpose of Transformation

Transformations are applied so that the data appear to more closely meet the assumptions of statistical inference procedure that is to be applied or to adjust/improve the interpretability of appearance of graphs. OR Transformations are chosen to stabilize variance and are commonly known as "variance-stabilizing transformations." The goal is to apply a power or other nonlinear transformation to the response variable, ensuring that fluctuations in outcomes remain consistent across different values of the independent variables. Residual plots are often used to help identify appropriate variance-stabilizing transformations by providing visual clues about the pattern of variability.

## How are they selected?

- If variance increases linearity, then we use the square root transformation on the response variable i,e sqrt(y)

- If variance increases then decreases parabolically, we use arcsin(sqrt(y))

- If variance increases quadratically, then we use ln(y)

## Descriptive statistics to investigate possible transformation of the response variable

Although formal methods exist for selecting appropriate transformations, one approach is to examine the shape of the residual plot(s) to make an educated guess about a suitable transformation. In the residual plot above, the variation appears to increase linearly with the predicted values, suggesting that a square root transformation of the response variable might be a reasonable option to try.

## Variance-stabilizing transformation of the outcome of interest

As stated above, the residual plot suggests that a square root transformation could be effective. Additionally, a logarithmic transformation might also be worth considering.

```{r transformation}
sqrt_accidents = sqrt(accidents)
```

## Using the transformed response, re-fit the linear model and evaluate the model assumptions

```{r Sqrt}
SQRTModel = lm(sqrt_accidents~exposure+construction1+construction2+construction3)
summary(SQRTModel)
```

<img width="566" alt="SQRT" src="https://github.com/user-attachments/assets/a6396c0f-7347-4e64-8a5f-7d9b8ff62285">

Hypothesis (overall F-test):

$H_0$: all slopes = 0

$H_A$: at least one slope ≠ 0

Conclusion: The overall F-test p-value is 6.219e-11, which is below 0.05, so the predictors are significantly related to √accidents. R² rises from 0.73 to 0.83, and the Shapiro-Wilk test on the residuals gives p = 0.554, with no evidence of non-normality.

```{r transformation normality }
library(car)
ncvTest(SQRTModel)
```

<img width="468" alt="Non CNS" src="https://github.com/user-attachments/assets/3227f8a9-0390-4f95-b3f4-e5bded75bacc">


Using the square root of the number of accidents as the new outcome variable, the test for non-constant variance shows no significant evidence of deviations from homoscedasticity (X² = 0.3813; p = 0.5369). The residual plot now resembles random noise, indicating that the transformation has addressed some model issues, including the previously observed changing variation in the outcome.

```{r non formal}
library(ggplot2)
qplot(SQRTModel$fitted.values,SQRTModel$residuals)
``` 

<img width="737" alt="SQgplot" src="https://github.com/user-attachments/assets/24c5c56b-5d59-42ea-90b1-decce65a459f">

## Process of weighted Least Squares to accommodate departures from constant variance

 - Fit the regression model by un-weighted least squares and analyze the residuals. 
 
 - Estimate the variance function or the standard deviation function by regressing either the squared residuals or the absolute residuals on the appropriate predictor(s). 
 
 - Use the fitted values from the estimated variance or standard deviation function to obtain the weights. 
 
 - Estimate the regression coefficients using these weights.

## Purpose of weighted Least Squares

Weighted Least Square is a remedy for non constant variance that is, it helps to reduce or eliminate
unequal variances of the error terms. That is, Weighted Least Squares (WLS) is a technique that adjusts both the estimation process and standard errors to account for varying levels of response variability across observations. Observations with higher variability are assigned "less weight" in parameter estimation and standard error calculations, while those with lower variability are given "greater weight". 

## What decision must be made by the researcher

The researcher has to choose the weights. In this approach, the researcher determines the weights for each observation, typically by applying a transformation to an independent variable in the model.

## Using descriptive statistics, investigate possible weights to be used in fitting the normal regression model

Scatterplots are commonly used to examine predictors linked to variation in the response variable. These plots typically display either the response values or residuals from a normal model against transformations of potential weighting variables. Shown are plots of residuals versus the square root of months in service and the square root of exposure, though other variables can also be explored.

```{r weights descriptive}
qplot(service_months,NormalModel$residuals)
qplot(service_months^(1/2),NormalModel$residuals)
```

```{r exposure descriptive}
qplot(exposure,NormalModel$residuals)
qplot(exposure^(1/2),NormalModel$residuals)
```

## Select and justify appropriate weights for fitting the proposed linear model.

```{r weights}
qplot(service_months^(1/2),NormalModel$residuals)
qplot(exposure^(1/2),NormalModel$residuals)
```

<img width="740" alt="serv" src="https://github.com/user-attachments/assets/a256675a-d4d8-4671-a828-a3bf248fa3bc">

<img width="717" alt="exp" src="https://github.com/user-attachments/assets/f8eec6ad-7c70-47df-8721-71be775d15ba">

Both predictors seem to explain some of the varying residual patterns. To address heteroscedasticity, the square root of the months in service will be used as weights in the regression analysis. 

## Using your weights, re-fit the linear model and evaluate the model assumptions.

```{r linearity of weights}
WModel = lm(accidents~exposure+construction1+construction2+construction3,weights=(service_months^(1/2)))
summary(WModel)
```

<img width="626" alt="Wmodell" src="https://github.com/user-attachments/assets/1c3a3521-113b-499c-b71e-426215833d4b">

## Re-evaluate model assumptions

- a. Non constant Variance

```{r weights non constant}
library(car)
ncvTest(WModel)
```

<img width="603" alt="NcvT" src="https://github.com/user-attachments/assets/27afe859-e4b1-4e8a-9fd1-f15a39a0e58c">

- b. Normality of variables

```{r weights normality}
shapiro.test(WModel$residuals)
```

- c. Linearity of regression function 

```{r linearity2}
library(ggplot2)
qplot(WModel$fitted.values,WModel$residuals)
```

<img width="732" alt="WmodelQplot" src="https://github.com/user-attachments/assets/d7df2ddc-9427-4628-8a8d-8bbb2597d0ed">

With √service_months as weights, the constant-variance test no longer detects significant heteroscedasticity (χ² = 1.3252, p = 0.2497), and the residuals pass the normality test (Shapiro-Wilk p = 0.080). However, the residual plot still suggests some heteroscedasticity.

**A caution about the weights:** In WLS, each weight should be proportional to 1 / Var(εᵢ). Weighting by √service_months assumes that ships with more service have *smaller* error variance. For accident counts, we would expect the reverse, because more time in service means larger and more variable counts. Using the inverse weights (1/√service_months) makes the test fail (p = 0.0015). The passing result above is therefore not strong evidence that WLS solved the problem.

## Conclusion

- The **square-root transformed model** is the best of the three. It meets the normality and constant-variance assumptions, and fit improves to R² = 0.83.
- `exposure` is the strongest predictor of ship accidents in every model.
- Because accidents are counts, the natural next model is a **Poisson or negative binomial GLM with log(service_months) as an offset**. A quick Poisson fit shows overdispersion (deviance/df ≈ 2.1), which favors negative binomial.
