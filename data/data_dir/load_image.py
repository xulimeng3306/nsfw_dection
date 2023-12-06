import requests
import os
import pandas as pd
from tqdm import tqdm


def download_image(url, save_path, save_path2):
    try:
        # 发送HTTP请求，获取图片数据
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求不成功，抛出异常

        # 获取文件名
        file_name = os.path.join(save_path, os.path.basename(url))
        file_name2 = os.path.join(save_path2, os.path.basename(url))

        # 保存图片到指定路径
        with open(file_name, 'wb') as file, open(file_name2, 'wb') as file2:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file2.write(chunk)

        # print(f"图片已保存到: {file_name}")
    except Exception as e:
        print(f"下载图片时发生错误: {str(e)}")


# 示例用法
# image_url = "https://example.com/image.jpg"  # 替换为实际的图片URL
# save_directory = "/path/to/save"  # 替换为实际的保存路径

# download_image(image_url, save_directory)

if __name__ == '__main__':
    file_path = "./正常.csv"
    save_path = "../ht_heads/0"
    all_image_path = "../all_images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(all_image_path):
        os.mkdir(all_image_path)
    data = pd.read_csv(file_path)
    for i in tqdm(data['url'].to_numpy()):
        download_image(i, save_path, all_image_path)
