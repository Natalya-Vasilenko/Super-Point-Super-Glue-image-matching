import os
import numpy as np
import cv2


def index_by_point(kp):
    d = {}
    index = 1
    for point in kp:
        p = tuple(point)
        if p not in d:
            d[p] = index
            index += 1
    return d


def matches_index(kp1, kp2, matches):
    ip1 = index_by_point(kp1)
    ip2 = index_by_point(kp2)
    return list(map(lambda match: [ip1[tuple(match[0])], ip2[tuple(match[1])]], matches))


def create_points_matches(keypoints1, keypoints2, matches):
    points_matches = []
    for i in range(matches.shape[0]):
        match = matches[i]
        points_matches.append([keypoints1[i], keypoints2[match]])
    return np.array(points_matches)


def delete_keypoints_duplicates(keypoints):
    s = []
    for kp in map(tuple, keypoints):
        if not kp in s:
            s.append(kp)
    return s


def prepeare_npz(file_name1, file_name2, npz):
    mask1 = cv2.imread(mask_path + '\\' + file_name1 + '.jpg.jpg').sum(axis=2)
    mask2 = cv2.imread(mask_path + '\\' + file_name2 + '.jpg.jpg').sum(axis=2)

    keypoints1 = npz['keypoints0']
    keypoints2 = npz['keypoints1']
    matches = npz['matches']

    points_matches = create_points_matches(keypoints1, keypoints2, matches)
    points_matches_in_mask = np.array(
        list(filter(lambda match: match_in_masks(mask1, mask2, match), points_matches))).astype(int)
    if points_matches_in_mask.shape[0] == 0:
        return
    keypoints1_in_mask = points_matches_in_mask[:, 0, :]
    keypoints2_in_mask = points_matches_in_mask[:, 1, :]
    matches = matches_index(keypoints1_in_mask, keypoints2_in_mask, points_matches_in_mask)
    return points_matches_in_mask
    save_keypoints(file_name1 + '.jpg.txt', delete_keypoints_duplicates(keypoints1_in_mask))
    save_keypoints(file_name2 + '.jpg.txt', delete_keypoints_duplicates(keypoints2_in_mask))
    with open('npz_pairs.txt', 'a') as file:
        write_matchings(file, file_name1 + '.jpg', file_name2 + '.jpg', matches)


def point_in_mask(mask, point):
    return mask[int(point[1]), int(point[0])] != 0


def match_in_masks(mask1, mask2, match):
    return point_in_mask(mask1, match[0]) & point_in_mask(mask2, match[1])


def filter_matrix_element_in_array(matrix, array, col=0):
    return np.array(list(filter(lambda x: x[col] in array, matrix)))


def indexes_in_mask(kp, mask):
    m_ = np.array(list(map(lambda point: point_in_mask(mask, point), kp)))
    return np.arange(len(kp))[m_]


def filter_matches_in_mask(kp1, kp2, mask1, mask2, matches):
    matches_ = matches.copy()

    indexes_in_mask1 = indexes_in_mask(kp1, mask1)
    indexes_in_mask2 = indexes_in_mask(kp2, mask2)

    matches_ = filter_matrix_element_in_array(matches_, indexes_in_mask1, col=0)
    matches_ = filter_matrix_element_in_array(matches_, indexes_in_mask2, col=1)

    return matches_


def save_keypoints(file_name, np_keypoints):
    with open(file_name, 'w') as file:
        file.write('')
    with open(file_name, 'a') as file:
        file.write(str(len(np_keypoints)) + ' 128''\n')
        for i in np_keypoints:
            file.write(str(i[0]) + ' ' + str(i[1]) + ' ' + ' '.join(map(str, [0.0, 0.0] + [0] * 128)) + '\n')


def write_matchings(file, img1_name, img2_name, np_pairs):
    file.write(img1_name + ' ' + img2_name + '\n')
    for i, j in np_pairs:
        file.write(str(i) + ' ' + str(j) + '\n')
    file.write('\n')


def save_matchings(file_name, img1_name, img2_name, np_pairs):
    with open(file_name, 'w') as file:
        file.write('')
    with open(file_name, 'a') as file:
        file.write(img1_name + ' ' + img2_name + '\n')
        for i, j in np_pairs:
            file.write(str(i) + ' ' + str(j) + '\n')

npz_path = r'C:\for_githab\SuperGlue_Lanit_dataset\SuperGluePretrainedNetwork-master\dump_match_pairs'
npz_files_names = os.listdir(npz_path)

mask_path = r'C:\for_githab\SuperGlue_Lanit_dataset\for_colmap\masks_jpg'
masks = os.listdir(mask_path)

with open('npz_pairs.txt', 'w') as file:
    file.write('')

for npz_file_name in npz_files_names:
    npz = np.load(npz_path + '\\' + npz_file_name)
    file_name1, file_name2, _ = npz_file_name.split('_')
    prepeare_npz(file_name1, file_name2, npz)
