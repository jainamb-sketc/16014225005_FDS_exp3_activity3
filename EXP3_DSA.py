import csv
import math
import os

def load_data(file_path):
    dataset = []
    if os.path.exists(file_path):
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                for cell in row:
                    try:
                        dataset.append(float(cell))
                    except ValueError:
                        continue
    return dataset

# Main Execution for Activity 3
if __name__ == "__main__":
    file_name = "london_weather_data_1979_to_2023.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    data = load_data(file_path)
    if not data:
        # Fallback dataset if file is absent
        data = [12.5, 15.2, 12.5, 18.1, 22.4, 20.0, 15.2, 12.5, 25.3, 30.1, 18.1, 20.0, 15.2, 14.8, 19.3, 22.4, 25.3, 11.2, 15.2, 18.1]

    # Create 4 equal class intervals
    min_val, max_val = min(data), max(data)
    num_classes = 4
    class_width = (max_val - min_val) / num_classes
    if class_width == 0:
        class_width = 1.0

    intervals = [(min_val + i * class_width, min_val + (i + 1) * class_width) for i in range(num_classes)]
    frequencies = [0] * num_classes

    # Count frequencies per interval
    for x in data:
        for i, (low, high) in enumerate(intervals):
            if i == num_classes - 1:
                if low <= x <= high:
                    frequencies[i] += 1
                    break
            else:
                if low <= x < high:
                    frequencies[i] += 1
                    break

    midpoints = [(low + high) / 2.0 for low, high in intervals]
    n = sum(frequencies)

    # Grouped Mean
    grouped_mean = sum(f * x for f, x in zip(frequencies, midpoints)) / n

    # Grouped Median
    cum_freq = []
    tot = 0
    for f in frequencies:
        tot += f
        cum_freq.append(tot)

    med_idx = 0
    for i, cf in enumerate(cum_freq):
        if cf >= n / 2.0:
            med_idx = i
            break

    l_med = intervals[med_idx][0]
    cf_prev = cum_freq[med_idx - 1] if med_idx > 0 else 0
    f_med = frequencies[med_idx]
    grouped_median = l_med + (((n / 2.0) - cf_prev) / f_med) * class_width if f_med > 0 else l_med

    # Grouped Mode
    modal_idx = frequencies.index(max(frequencies))
    l_mode = intervals[modal_idx][0]
    f1 = frequencies[modal_idx]
    f0 = frequencies[modal_idx - 1] if modal_idx > 0 else 0
    f2 = frequencies[modal_idx + 1] if modal_idx < num_classes - 1 else 0
    denom = (2 * f1 - f0 - f2)
    grouped_mode = l_mode + ((f1 - f0) / denom) * class_width if denom != 0 else l_mode

    # Grouped Variance & Std Dev
    sq_sum = sum(f * ((x - grouped_mean) ** 2) for f, x in zip(frequencies, midpoints))
    grouped_variance = sq_sum / (n - 1)
    grouped_std_dev = math.sqrt(grouped_variance)

    # Grouped Quartiles and IQR
    def get_grouped_quantile(q_pos):
        idx = 0
        for i, cf in enumerate(cum_freq):
            if cf >= q_pos:
                idx = i
                break
        l_q = intervals[idx][0]
        cf_p = cum_freq[idx - 1] if idx > 0 else 0
        f_q = frequencies[idx]
        return l_q + ((q_pos - cf_p) / f_q) * class_width if f_q > 0 else l_q

    q1_grouped = get_grouped_quantile(n * 0.25)
    q3_grouped = get_grouped_quantile(n * 0.75)
    grouped_iqr = q3_grouped - q1_grouped

    # Print Report
    print("=========================================================")
    print("        ACTIVITY 3: GROUPED DATA COMPUTATION")
    print("=========================================================")
    print(" Class Interval     | Midpoint (xi) | Frequency (fi) | Cumulative Freq")
    print(" ---------------------------------------------------------------------")
    for i in range(num_classes):
        low, high = intervals[i]
        print(f" [{low:6.2f} - {high:6.2f})  |   {midpoints[i]:8.2f}    |       {frequencies[i]:2d}       |       {cum_freq[i]:2d}")
    print(" ---------------------------------------------------------------------")
    print(f" Grouped Mean          : {grouped_mean:.4f}")
    print(f" Grouped Median        : {grouped_median:.4f}")
    print(f" Grouped Mode          : {grouped_mode:.4f}")
    print(f" Grouped Variance      : {grouped_variance:.4f}")
    print(f" Grouped Std Deviation : {grouped_std_dev:.4f}")
    print(f" Grouped Q1            : {q1_grouped:.4f}")
    print(f" Grouped Q3            : {q3_grouped:.4f}")
    print(f" Grouped IQR           : {grouped_iqr:.4f}")
    print("=========================================================")