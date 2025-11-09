# EEG Event Detection Dataset Overview

There are **12 subjects** in total, **10 series of trials** for each subject, and approximately **30 trials** within each series. The number of trials varies for each series.

## Train / Test Split

- The **training set** contains the **first 8 series** for each subject.
- The **test set** contains the **9th and 10th series**.

## Target Events

For each GAL, you are tasked with detecting the following **6 events**:

1. HandStart
2. FirstDigitTouch
3. BothStartLoadPhase
4. LiftOff
5. Replace
6. BothReleased

These events always occur in the same order.

## Files in the Training Set

In the training set, there are **two files** for each **subject + series** combination:

- `*_data.csv`  
  Contains the raw **32-channel EEG data** sampled at **500 Hz**.
- `*_events.csv`  
  Contains the **ground truth frame-wise labels** for all events.

The event files for the **test set** are **not provided** and must be predicted.

Each timeframe is given a unique `id` column according to the **subject**, **series**, and **frame** to which it belongs.

The six label columns are either:

- `0` if the corresponding event has **not** occurred within **±150 ms** (**±75 frames**), or
- `1` if the corresponding event **has** occurred within that window.

A perfect submission will predict a probability of **1** for this entire window.

## Important Note

When predicting, you may **not use data from the future**.

For example, if you are predicting labels for `subj1_series9_11`, you may **not incorporate any frame after 11** of series 9 from subject 1.

In a real-world detection algorithm, future data does not exist.

## Data Leakage Warning

You must be careful not to include **data leakage**.

For example, you may **not center the signals** for `subj1_series9_11` based on the mean of **all frames** for `subj1_series9`. Instead, you must use the mean based only on **frame 0 to 10**.

The host will be checking models to ensure they do **not violate this rule**.

It is recommended to structure your code so that it is abundantly clear that you are **not accessing future data** (or features based on future data) when predicting.

## Training on Other Subjects / Series

You may train on other subjects and series outside of the series for which you are predicting.

For example, you can use **all of `subj2_series6`** when predicting **`subj1_series5`**.

## Electrode Channel Information

The columns in the data files are labeled according to their associated electrode channels.

You may make use of the spatial relationship between the electrode locations, as shown in the referenced diagram.

> Note: The electrode-location diagram itself was not included in the provided text, so it is not embedded in this markdown file.
