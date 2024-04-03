import numpy as np

def gen_data(n_examples=1000,
             start_low=0, start_high=10,
             delta_low=1, delta_high=5,
             len_low=5, len_high=30):

    data = []
    for n in range(n_examples):
        start = np.random.randint(low=start_low, high=start_high)
        delta = np.random.randint(low=delta_low, high=delta_high)
        length = np.random.randint(low=len_low, high=len_high)

        d = start + np.arange(length) * delta
        data.append(d)

    return data
        
    
