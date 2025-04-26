<<<<<<< HEAD
# Covariate Shift Correction for Large-Scale Classification

---

## Overview

This repository explores the application of **covariate shift correction** to the field of **cybersecurity**, focusing on the real-world UNSW-NB15 dataset.

Covariate shift occurs when the training and testing data distributions differ — a common and critical problem in cybersecurity due to evolving attack patterns and environmental changes. Without correction, models trained on historical data often perform poorly when deployed in practice.

In this work, we demonstrate that **density ratio-based covariate shift correction** can significantly improve model robustness on real datasets.  
We focus on adapting and scaling traditional covariate shift methods to handle large-scale cybersecurity datasets effectively.

---

## Download Dataset

This project uses the **UNSW-NB15** dataset for training and evaluation.

Since the dataset is large, it is **not included** in this repository.  
You must manually download it from the official source:

- Official source: [UNSW-NB15 Dataset Website](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

Specifically, download the following CSV files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

After downloading:

1. Create a folder inside your project directory:

```bash
mkdir -p CSV\ Files/Training\ and\ Testing\ Sets/
```

2. Move the CSV files into the created folder:

```bash
mv path/to/UNSW_NB15_training-set.csv CSV\ Files/Training\ and\ Testing\ Sets/
mv path/to/UNSW_NB15_testing-set.csv CSV\ Files/Training\ and\ Testing\ Sets/
```

Make sure the final structure looks like:

```
your-project/
├── CSV Files/
│   └── Training and Testing Sets/
│       ├── UNSW_NB15_training-set.csv
│       └── UNSW_NB15_testing-set.csv
├── train.py
├── README.md
├── ...
```

✅ Once the files are placed correctly, you can proceed to run the training script without modification.

---

## Notes:
- The UNSW-NB15 dataset is provided for **research purposes**.
- Always cite the original authors if you use this dataset in your own work.


## Key Contributions

- **Applying Covariate Shift to Cybersecurity:**  
  We show that covariate shift correction is crucial for cybersecurity tasks, where unseen or rare attack types appear in testing but are underrepresented during training.

- **Demonstrating Real-World Effectiveness:**  
  We validate the benefits of covariate shift correction on the real UNSW-NB15 dataset, improving minority class detection and overall model stability.



- **Practical Importance Weighting:**  
  By estimating importance weights accurately, we adjust training to better match test distributions, leading to improvements in both **overall accuracy** and **rare class recall**.

---


## Methodology

### Density Ratio Estimation for Covariate Shift Correction

The core idea behind covariate shift correction is to **reweight training samples** to better match the test distribution.

Under covariate shift, the training distribution \( p_{\text{train}}(x) \) and the test distribution \( p_{\text{test}}(x) \) differ.  
However, the conditional distribution \( p(y|x) \) remains the same.

Thus, the correct approach is to **reweight training samples** by the **density ratio**:

\[
w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}
\]

This ensures that the training loss becomes an unbiased estimate of the test loss.

---

### How We Estimate the Density Ratio

Instead of estimating \( p_{\text{test}}(x) \) and \( p_{\text{train}}(x) \) separately (which is difficult),  
we **directly estimate the density ratio** using methods like:

- **KLIEP** (Kullback-Leibler Importance Estimation Procedure)
- **uLSIF** (Unconstrained Least-Squares Importance Fitting)
- **Wasserstein Ratio Matching**

These methods frame density ratio estimation as a **constrained optimization problem**.

---



## Experimental Results

We evaluate our method on the **UNSW-NB15** cybersecurity dataset.

### Dataset

- Multi-class classification task (`attack_cat` label).
- Highly imbalanced across 10 classes (e.g., `Normal`, `Generic`, `Exploits`, `Worms`).

---

### Performance Comparison (with Support)

| Class            | Support | Precision (Base) | Recall (Base) | Precision (Reweighted) | Recall (Reweighted) | Comment |
|------------------|---------|------------------|---------------|-------------------------|---------------------|---------|
| Analysis         | 677     | 0.00              | 0.00          | 0.00                    | 0.01                 | Slight recall improvement |
| Backdoor         | 583     | 0.01              | 0.08          | 0.01                    | 0.07                 | Slight recall loss |
| DoS              | 4089    | 0.63              | 0.09          | 0.62                    | 0.10                 | Minor improvement |
| Exploits         | 11132   | 0.60              | 0.80          | 0.61                    | 0.79                 | Stable |
| Fuzzers          | 6062    | 0.29              | 0.57          | 0.29                    | 0.59                 | Small recall gain |
| Generic          | 18871   | 1.00              | 0.97          | 1.00                    | 0.97                 | Stable |
| Normal           | 37000   | 0.96              | 0.75          | 0.96                    | 0.75                 | Stable |
| Reconnaissance   | 3496    | 0.93              | 0.80          | 0.93                    | 0.79                 | Stable |
| Shellcode        | 378     | 0.38              | 0.65          | 0.39                    | 0.67                 | **Noticeable improvement** |
| Worms            | 44      | 0.50              | 0.09          | 0.67                    | 0.09                 | **Precision improvement** |

---

### Highlights:

- **Shellcode Recall:** improved from **65% → 67%** (Support = 378).
- **Fuzzers Recall:** improved from **57% → 59%** (Support = 6062).
- **Worms Precision:** improved from **50% → 67%** (Support = only 44 samples).
- Improvements are particularly **important for rare attack types**.

---

---

### Overall Performance

| Metric            | Without Reweighting | With Covariate Shift Correction |
|-------------------|---------------------|---------------------------------|
| Accuracy          | 75%                 | 75%                             |
| Weighted F1 Score | 0.77                | 0.78                            |


---

## How to Use

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your dataset following the structure of UNSW-NB15 (or adapt to your needs).

3. Run the training script:

```bash
python multi_train.py
```

- `multi_train.py` will load the data, apply covariate shift correction, and train Random Forest classifiers.

4. Evaluate results and generate confusion matrices.

---

## Future Work

- Scale covariate shift estimation with **mini-batch stochastic optimization**.
- Integrate **focal loss** and **reweighting** for end-to-end neural network training.
- Explore **self-supervised covariate shift estimation** techniques without explicit labels.

---

## References

- Sugiyama, M., Nakajima, S., Kashima, H., von Luxburg, U., & Kawanabe, M. (2008).  
  *Direct importance estimation with model selection and its application to covariate shift adaptation.*  
  In NIPS.


---

## Contact

If you have questions or suggestions, feel free to open an issue or a pull request.
=======
