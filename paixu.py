def read_order_from_file(order_file_path):
    """
    从给定的文件中读取视频文件的顺序。
    :param order_file_path: 包含视频文件名顺序的文件路径。
    :return: 按照文件中定义顺序的视频文件列表。
    """
    with open(order_file_path, 'r', encoding='utf-8') as file:
        video_order = [line.strip() for line in file.readlines()]

    return video_order


def read_and_sort_txt_by_video_names(txt_file_path, sorted_video_names):
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 创建一个字典来存储原始txt文件中的每一行数据
    video_scores_dict = {line.split(',')[0].strip(): line.strip() for line in lines}

    # 根据sorted_video_names重新组织数据
    sorted_lines = []
    for video_name in sorted_video_names:
        if video_name in video_scores_dict:
            sorted_lines.append(video_scores_dict[video_name])
        else:
            print(f"警告: 视频 {video_name} 没有对应的评分记录")

    return sorted_lines


# 文件夹路径和txt文件路径
order_file_path = "/thqa_ntire_testlist.txt"  # 手动创建的顺序文件路径
txt_file_path = "outputs.txt"
output_txt_file_path = "output.txt"

# 获取已排序的视频文件名列表（基于手动指定的顺序）
sorted_video_names = read_order_from_file(order_file_path)

# 根据文件夹中的视频文件顺序重新排列txt文件内容
sorted_lines = read_and_sort_txt_by_video_names(txt_file_path, sorted_video_names)

# 将重新排序后的内容写入新的txt文件
with open(output_txt_file_path, 'w', encoding='utf-8') as file:
    for line in sorted_lines:
        file.write(line + '\n')

print(f"已根据文件夹中的视频顺序整理完毕，结果保存在 {output_txt_file_path}")