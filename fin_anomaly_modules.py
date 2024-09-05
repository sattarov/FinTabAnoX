import numpy as np
import torch
import pandas as pd



# Function to add different types of noise into clean data and synthetically generate outlier samples
def add_noise(
        clean_data, 
        cat_features, 
        num_features, 
        cat_attr_positions, 
        num_attr_positions, 
        pct_outliers,
        rng,
        num_noise_max_fraction=0.5, 
        cat_noise_max_fraction=0.5
    ):
    """ Method for Data Corruption on the fly.
    :param clean_data: Input data with n samples and m features, size n xm
    :param cat_features: List of Categorical attribute column name from the Input data
    :param num_features: List of Numerical attribute column name from the Input data
    :param cat_attr_positions: Dictionary of Categorical attribute column name and its position in the Input data
    :param num_attr_positions: Dictionary of Numerical attribute column name and its position in the Input data
    :param pct_outliers: Percentage of outliers to be generated
    :param rng: Random number generator
    :param num_noise_max_fraction: Fraction of numerical columns to be corrupted in a row
    :param cat_noise_max_fraction: Fraction of categorical columns to be corrupted in a row
    :return:
            noise_data: Data with few corrupted samples. n samples and m features, size n xm
    """

    assert len(clean_data) > 0, "Empty input data.."
    
    # percentage of outliers to be generated
    n_outliers=int(len(clean_data)*pct_outliers)

    noise_data = clean_data.clone()

    ########## start adding noise by selecting random rows and columns from the input data ##########
    # pick random row indices
    noise_row_ids = rng.choice(range(len(clean_data)), n_outliers, replace=False)
    # y_anomaly = np.zeros(len(clean_data))
    # y_anomaly[random_rows] = 1
    min_num_cols_noise = 1
    max_num_cols_noise = int(len(num_features) * num_noise_max_fraction)
    min_cat_cols_noise = 1  # 1 if random_cols_size_num == 0 else 0
    max_cat_cols_noise = int(len(cat_features) * cat_noise_max_fraction)

    # get the max and min position index of numerical attributes
    num_attr_position_start = list(num_attr_positions.values())[0]
    num_attr_position_end = list(num_attr_positions.values())[-1]

    # random_cols_num = []
    for random_row in noise_row_ids:

        ############## pick random numerical column indices
        random_cols_size_num = rng.integers(low=min_num_cols_noise, high=max_num_cols_noise, endpoint=True)
        random_cols_idx = rng.choice(a=np.arange(num_attr_position_start, num_attr_position_end), size=random_cols_size_num, replace=False)

        ### sample random noise type
        numnoise_type = rng.integers(low=1, high=3, endpoint=True)

        # Gaussian noise
        if numnoise_type == 1:
            random_std_scale = rng.integers(low=3, high=5, endpoint=True, size=random_cols_size_num)
            noise_val = rng.normal(loc=0, scale=random_std_scale, size=random_cols_size_num)

        # Laplace noise
        elif numnoise_type == 2:
            random_std_scale = rng.integers(low=3, high=5, endpoint=True, size=random_cols_size_num)
            noise_val = rng.laplace(loc=0, scale=random_std_scale, size=random_cols_size_num)

        # LogNormal noise
        elif numnoise_type == 3:
            # random_std_scale = np.random.choice(a=range(1, 16))
            sigma = 1.0
            mu = 0.01
            log_mu = np.log(mu)
            normal_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
            normal_mean = log_mu - normal_std ** 2 / 2
            noise_val = rng.lognormal(mean=normal_mean, sigma=normal_std, size=random_cols_size_num)
        else:
            raise ValueError('Wrong numeric type defined')

        ### add noise entry
        noise_data[random_row, random_cols_idx] += torch.tensor(noise_val, device=noise_data.device)
        # clean_data.index_add_(dim=0, index=torch.tensor(random_cols_idx), source=torch.tensor(noise_val))

        ############ pick random categorical column indices
        random_cols_size_cat = rng.integers(low=min_cat_cols_noise, high=max_cat_cols_noise, endpoint=True)
        # random_cols_size_cat = rng.choice(a=range(min_cat_cols_noise, max(2, len(cat_features)//2)))
        random_cols = rng.choice(a=cat_features, size=random_cols_size_cat, replace=False)
        for random_col in random_cols:
            random_col_position = cat_attr_positions[random_col]
            # get current value
            current_value = clean_data[random_row, random_col_position].cpu().numpy()
            current_value_not_active_position = np.argwhere(current_value == 0).flatten()
            noise_position = rng.choice(current_value_not_active_position)
            # replace
            noise_val = current_value * 0.0
            noise_val[noise_position] = 1.0
            noise_data[random_row, random_col_position] = torch.tensor(noise_val, device=noise_data.device)

    return noise_data


def precision_at_K(y_scores, y_true):
    """ Compute precision at K
    :param y_scores: Reconstruction errors
    :param y_true: True labels
    :return: precision at K
    """
    # determine the threshold K
    K = sum(y_true)

    # convert to pd series
    y_true = pd.Series(y_true, name='LABEL', index=range(len(y_true)))
    y_scores = pd.Series(y_scores, name='SCORES', index=range(len(y_scores)))

    # combine labels and scores
    y_values  = pd.concat([y_true, y_scores], axis=1)

    # sort scores in descending order
    y_values.sort_values('SCORES', ascending=False, inplace=True)

    # select top K
    top_K = y_values.iloc[:K]
    
    # compute precision at K
    precision = top_K['LABEL'].sum() / K
    return precision