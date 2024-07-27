import os
import requests

data_folder = "../data/mobilityflows"
os.makedirs(data_folder, exist_ok=True)

for i in range(20):
    url = f"https://raw.githubusercontent.com/GeoDS/COVID19USFlows-WeeklyFlows-Ct2021/master/weekly_flows/ct2ct/2021_01_04/weekly_ct2ct_2021_01_04_{i}.csv"
    response = requests.get(url)
    file_path = os.path.join(data_folder, f"weekly_ct2ct_2019_06_10_{i}.csv")
    with open(file_path, 'wb') as file:
        file.write(response.content)

print(f"Downloaded files to {data_folder}:")
print(os.listdir(data_folder))
print(f"Download complete.")