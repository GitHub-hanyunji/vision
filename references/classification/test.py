import os

# 하위 디렉토리의 경로
directory_path = 'C:\\Users\AIRLAB\\바탕 화면\\airLab\\Projects\\dataset2'

# 하위 디렉토리들만 얻기
subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

print(subdirs)


# 하위 디렉토리의 경로
directory_path1 = r'C:\\Users\AIRLAB\\바탕 화면\\airLab\\Projects\\dataset1\\test'

# 하위 파일들의 개수
file_count = len([f for f in os.listdir(directory_path1) if os.path.isfile(os.path.join(directory_path1, f))])

print(f"하위 파일 개수: {file_count}")
