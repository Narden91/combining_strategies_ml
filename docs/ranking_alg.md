# Ranking Method

The ranking method uses a simpler approach based on:

1. **Pairwise Disagreement Matrix**
- Measures basic disagreement between pairs of classifiers
- Focuses on incorrect predictions only
- Creates a symmetric matrix of common errors

1. **Task Selection**
- Uses validation accuracies and disagreement information
- Fixed weighting between diversity and accuracy
- Simple pool creation through sequential selection

## Differences betweeen ranking algorithms

1. **Simpler Diversity Measurement**
   - Only considers disagreements between classifiers
   - Doesn't use multiple diversity metrics
   - No adaptive weighting based on prediction patterns

2. **Fixed Weighting**
   - Uses a constant 50-50 split between diversity and accuracy
   - Doesn't adapt weights based on data characteristics
   - More rigid ensemble selection process

3. **Pool Selection**
   - Creates final pool based on fixed threshold
   - Doesn't consider incremental contribution of each member
   - Less sophisticated than the greedy selection in entropy method

4. **Common Error Matrix Creation**
```python
T1 -> [Wrong on instances: 3, 5, 8, 10]
T2 -> [Wrong on instances: 2, 7, 10]
Common errors between T1 and T2 = 1 (instance 10)
```

1. **Diversity Score Calculation**
- Sum up common errors for each classifier
- Normalize by maximum value
- Lower score = more diverse (fewer common errors)

1. **Combination with Accuracy**
```python
Final Score = 0.5 * Diversity_Score + 0.5 * Validation_Accuracy
```

1. **Pool Selection**
- Sort by combined score
- Take top N classifiers
- Use majority voting for final prediction

Weaknesses of Original Method:
1. Doesn't consider prediction patterns
2. Fixed weighting scheme
3. Less adaptable to different datasets
4. Simpler diversity measurement

Strengths of Original Method:
1. Computationally simpler
2. More interpretable
3. Less sensitive to parameter tuning
4. Works well with validation accuracy information
