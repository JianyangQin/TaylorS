import csv
import numpy as np

if __name__ == '__main__':
    adj_filename = '../datasets/PEMS03/PEMS03_origin.csv'
    node_filename = '../datasets/PEMS03/PEMS03.txt'

    # 用于保存每一行文本的列表
    nodes_list = []

    # 逐行读取文本文件并保存到列表中
    with open(node_filename, 'r') as file:
        for line in file:
            # 去除每行末尾的换行符并添加到列表中
            nodes_list.append(int(line.strip()))

    nodes_list = np.asarray(nodes_list)

    origin, destination, cost = [], [], []
    with open(adj_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            org, des, c = np.where(nodes_list==int(row[0]))[0][0], np.where(nodes_list==int(row[1]))[0][0], float(row[2])
            origin.append(org)
            destination.append(des)
            cost.append(c)

    # 指定CSV文件名
    csv_file = '../datasets/PEMS03/PEMS03.csv'

    # 将三个列表的数据写入CSV文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头（可选）
        writer.writerow(['from', 'to', 'cost'])

        # 将数据写入CSV文件
        for i in range(len(origin)):
            writer.writerow([origin[i], destination[i], cost[i]])

    print(f'Data has been written to {csv_file}.')

