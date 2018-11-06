import numpy as np
import pandas as pd


def combined_dataset(file_1, file_2):
    different_pairs = pd.read_csv(file_1)
    same_pairs = pd.read_csv(file_2)
    frames = [different_pairs, same_pairs]
    return pd.concat(frames)


def feature_subtracted_dataset(combined_file, input_feature_data_file, is_gsc=False):
    feature_pairs = pd.read_excel(combined_file)
    data = pd.read_csv(input_feature_data_file)
    difference = []
    if is_gsc:
        diff_pair = feature_pairs.iloc[0:762557]
        same_pair = feature_pairs.iloc[762558:]
        feature_pairs = pd.concat([pd.DataFrame.sample(diff_pair, n=10000), pd.DataFrame.sample(same_pair, n=10000)])
    else:
        diff_pair = feature_pairs.iloc[0:293031]
        same_pair = feature_pairs.iloc[293032:]
        feature_pairs = pd.concat([pd.DataFrame.sample(diff_pair, n=791), pd.DataFrame.sample(same_pair, n=791)])
    subtracted_dataset = pd.DataFrame()
    for index, row in feature_pairs.iterrows():
        if is_gsc:
            f1 = data.loc[data['img_id'] == row[0]].iloc[0, 1:]
            f2 = data.loc[data['img_id'] == row[1]].iloc[0, 1:]
        else:
            f1 = data.loc[data['img_id'] == row[0]].iloc[0, 2:]
            f2 = data.loc[data['img_id'] == row[1]].iloc[0, 2:]
        row_values = []
        row_values.append(row[0])
        row_values.append(row[1])
        diff = f1.sub(f2, axis=0).abs()
        row_values.extend(diff)
        row_values.append(row[2])
        difference.append(row_values)
    subtracted_dataset = subtracted_dataset.append(difference)
    return subtracted_dataset


def feature_concatenated_dataset(combined_file, input_data_file, is_gsc=False):
    feature_pairs = pd.read_excel(combined_file)
    data = pd.read_csv(input_data_file)
    concat = []
    if is_gsc:
        diff_pair = feature_pairs.iloc[0:762557]
        same_pair = feature_pairs.iloc[762558:]
        feature_pairs = pd.concat([pd.DataFrame.sample(diff_pair, n=10000), pd.DataFrame.sample(same_pair, n=10000)])
    else:
        diff_pair = feature_pairs.iloc[0:293031]
        same_pair = feature_pairs.iloc[293032:]
        feature_pairs = pd.concat([pd.DataFrame.sample(diff_pair, n=791), pd.DataFrame.sample(same_pair, n=791)])
    for index, row in feature_pairs.iterrows():
        if is_gsc:
            f1 = data.loc[data['img_id'] == row[0]].iloc[0, 1:]
            f2 = data.loc[data['img_id'] == row[1]].iloc[0, 1:]
        else:
            f1 = data.loc[data['img_id'] == row[0]].iloc[0, 2:]
            f2 = data.loc[data['img_id'] == row[1]].iloc[0, 2:]
        row_values = []
        row_values.append(row[0])
        row_values.append(row[1])
        total_val = pd.concat([f1, f2], axis=0, ignore_index=True)
        row_values.extend(total_val)
        row_values.append(row[2])
        concat.append(row_values)
    subtracted_dataset = pd.DataFrame(concat)
    return subtracted_dataset


if __name__ == '__main__':
    dataset = combined_dataset("./HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv",
                               "./HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv")
    dataset.to_excel("./HumanObserved-Dataset/HumanObserved-Features-Data/combined_dataset.xlsx", 'Sheet1')
    print(dataset.shape)

    dataset = combined_dataset("./GSC-Dataset/GSC-Features-Data/diffn_pairs.csv",
                               "./GSC-Dataset/GSC-Features-Data/same_pairs.csv")
    dataset.to_excel("./GSC-Dataset/GSC-Features-Data/combined_dataset.xlsx", 'Sheet1')
    print(dataset.shape)
    import datetime
    subtracted_dataset = feature_subtracted_dataset(
        "./HumanObserved-Dataset/HumanObserved-Features-Data/combined_dataset.xlsx",
        "./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv")
    subtracted_dataset.to_csv("./HumanObserved-Dataset/HumanObserved-Features-Data/subtracted_dataset.csv")
    print("Done with Subtracted Human")

    start = datetime.datetime.now()
    subtracted_dataset = feature_subtracted_dataset("./GSC-Dataset/GSC-Features-Data/combined_dataset.xlsx",
                                                    "./GSC-Dataset/GSC-Features-Data/GSC-Features.csv", is_gsc=True)
    subtracted_dataset.to_csv("./GSC-Dataset/GSC-Features-Data/subtracted_dataset.csv")
    print("Done with Subtracted GSC")
    print(datetime.datetime.now()-start)

    concatenated_dataset = feature_concatenated_dataset(
        "./HumanObserved-Dataset/HumanObserved-Features-Data/combined_dataset.xlsx", "./HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv")
    concatenated_dataset.to_csv("./HumanObserved-Dataset/HumanObserved-Features-Data/concatenated_dataset.csv")
    print("Done with Concatenated Human")
    start = datetime.datetime.now()
    concatenated_dataset = feature_concatenated_dataset("./GSC-Dataset/GSC-Features-Data/combined_dataset.xlsx", "./GSC-Dataset/GSC-Features-Data/GSC-Features.csv", is_gsc=True)
    concatenated_dataset.to_csv("./GSC-Dataset/GSC-Features-Data/concatenated_dataset.csv")
    print("Done with concatenated GSC")
    print(datetime.datetime.now()-start)