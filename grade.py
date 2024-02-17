# importing required libraries
from PIL import Image
import numpy as np
import sys
import math
import copy



def gaussian_kernel_builder(area, standard_deviation):
    #  Creates a Gaussian kernel for image processing.
    kernel = np.zeros((area, area))
    for i in range(area):
        index_of_first_coordinate = int(abs(area / 2 + 0.5 - i - 1))
        gaussian_distr_first = Finding_gaussian_distribution(index_of_first_coordinate, standard_deviation)
        for j in range(area):
            index_of_second_coordinate = int(abs(area / 2 + 0.5 - j - 1))
            gaussian_distribution_second = Finding_gaussian_distribution(index_of_second_coordinate, standard_deviation)
            kernel[i, j] = (gaussian_distr_first ** 2 + gaussian_distribution_second ** 2) ** 0.5

    return kernel / np.sum(kernel)

def kernel_aspects(area, standard_deviation):
    # This Function is used to create a kernel which gives importance to certain aspects for image processing.
    matrix__of_gaussian_kernal = gaussian_kernel_builder(area, standard_deviation)
    matrix_of_identity_kernel = np.zeros((area, area))
    matrix_of_identity_kernel[int(area / 2), int(area / 2)] = 1
    kernel_of_aspects = matrix_of_identity_kernel - matrix__of_gaussian_kernal
    return kernel_of_aspects


def kernel_option_finder():
    # Creates a kernel for detecting answer options on the answer sheet.
    kernel = np.zeros((box_height_of_option_choice, box_width_of_option_choice))
    kernel[:2, :] = -1
    kernel[:, :2] = -1
    kernel[:, -2:] = -1
    kernel[-2:, :] = -1
    return kernel



def Finding_gaussian_distribution(input_value, standard_deviation):
    """This function helps us to calculate the value of Gaussian distribution at a particular input value point."""
    # formula used to acheive this calculation is e**((-((input_value)/(standard_deviation)))*2)/2)/standard_deviation
    return math.exp(-((input_value / standard_deviation) ** 2) / 2) / standard_deviation



def kernel_sharpening(area, standard_deviation):
    #  Creates a kernel to sharpen the image using kernel aspects.
    kernel_of_aspects = kernel_aspects(area, standard_deviation)
    matrix_of_identity_kernel = np.zeros((area, area))
    matrix_of_identity_kernel[int(area / 2), int(area / 2)] = 1
    sp_kernel = kernel_of_aspects + matrix_of_identity_kernel
    return sp_kernel



def bands_kernel(arr_of_image, kernel, starting_channel=True):
    # applies kernel to all bands of the image.
    if len(arr_of_image.shape) == 2:
        return kernel_worker(arr_of_image, kernel)
    dim_of_band = 0 if starting_channel else -1
    result = None  
    for i in range(arr_of_image.shape[0] if starting_channel else arr_of_image.shape[-1]):
        channel_arr = arr_of_image[i] if starting_channel else arr_of_image[..., i]
        if i != 0:
            result = np.concatenate((result, np.expand_dims(kernel_worker(channel_arr, kernel), axis=dim_of_band)),
                                    axis=dim_of_band)
        else:
            result = np.expand_dims(kernel_worker(channel_arr, kernel), axis=dim_of_band)
    return result


def kernel_worker(arr_of_image, kernel):
    # This function does the application of a kernel to an image represented as a NumPy array.
    solution = np.zeros((arr_of_image.shape[0] + 1 - kernel.shape[0], arr_of_image.shape[1] + 1 - kernel.shape[1]))
    for i in range(arr_of_image.shape[0] + 1 - kernel.shape[0]):
        for j in range(arr_of_image.shape[1] + 1 - kernel.shape[1]):
            solution[i, j] = np.sum(arr_of_image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return solution



def dictionary_values(x_coordinates, y_coordinates):
    # Generate a dictionary from given x and y coordinates
    value_dictionary = {}  
    for x, y in zip(x_coordinates, y_coordinates):
        if x not in value_dictionary:
            value_dictionary[x] = {}
        if y not in value_dictionary[x]:
            value_dictionary[x][y] = 1
    return value_dictionary


def surroundings_checker(value_dictionary, x_coordinate, y_coordinate, search_range, search=True):

    # Checks if neighboring coordinates around (x_coordinate, y_coordinate) within the given range exist in the dictionary.

    for i in range(x_coordinate - search_range, x_coordinate + search_range + 1):
        
        for j in range(y_coordinate - search_range, y_coordinate + search_range + 1):
            if search and i == x_coordinate and j == y_coordinate:
                continue
            if i in value_dictionary and j in value_dictionary[i]:
                return True
    return False


def get_x_y_from_flatten_arr(indices, shape):
    # This function is used to convert the indices of a flattened array back into their corresponding 2D coordinates.
    x_part = (indices / shape[1]).astype(int)
    y_part = indices - x_part * shape[1]
    return x_part, y_part





def horizontal_points(start_x, value_dictionary):
    # gets all points horizontally adjacent to the given starting x-coordinate from the value dictionary.
    points = []
    for x_coord in range(start_x, start_x + 10):
        if x_coord in value_dictionary:
            for y_coord in value_dictionary[x_coord]:
                points.append((x_coord, y_coord))
    return points

def question_or_not(start_x, start_y, all_points):
    
    # Determines if a point is part of a question area based on its coordinates and nearby points.

    total_points_found = 0  
    total_width_error = 0  
    width_of_y = spacing_of_option_box_width  
    option_points = [(start_x, start_y, True)]  

    previous_point_y = start_y + width_of_y  
    for _ in range(4):
        point_found = False  
        for point in all_points:
            if (previous_point_y - 4) <= point[1] <= (previous_point_y + 4):
                # Update total width error
                total_width_error += abs(point[1] - previous_point_y)
                previous_point_y = point[1] + width_of_y  
                total_points_found += 1  
                point_found = True  
                option_points.append((point[0], point[1], True))  
                break
        if not point_found:
            option_points.append((start_x, previous_point_y, False))
            previous_point_y += width_of_y

    is_question = (total_points_found >= threshold_point_error_question) and ((total_width_error / total_points_found) <= threshold_width_avg_error_question)
    return is_question, option_points  


def top_k_not_close_to_each_other(x_values, y_values, k_value, close_distance=10):
    
    # Extract the top k points that are not in close proximity to each other.
    values_dict = dictionary_values(x_values, y_values)

    removed_x_coords, removed_y_coords = [], []
    filtered_x_coords, filtered_y_coords = [], []
    for x, y in zip(x_values, y_values):
        if surroundings_checker(values_dict, x, y, close_distance):
            removed_x_coords.append(x)
            removed_y_coords.append(y)
            del values_dict[x][y]
            continue
        filtered_x_coords.append(x)
        filtered_y_coords.append(y)

        if len(filtered_y_coords) == k_value:
            break
    return np.array(filtered_x_coords), np.array(filtered_y_coords), np.array(removed_x_coords), np.array(removed_y_coords)




def line_above(x_values, y_values, x_threshold=0):
    # Find the top line of a set of points based on their coordinates.

    value_dict = dictionary_values(x_values, y_values)
    min_x = min(x_values) 

    points = []
    for x in range(min_x, min_x + 5):
        if x in value_dict:
            for y in value_dict[x]:
                points.append((x, y))  

    x_errors = {x: 0 for x in range(min_x, min_x + 5)}
    for x in range(min_x, min_x + 5):
        for point in points:
            x_errors[x] += (point[0] - x) ** 2  

    min_x_err = min_x  
    for x in x_errors:
        if x_errors[x] < x_errors[min_x_err]:
            min_x_err = x   





def is_near(x_values, y_values):
    # Count the number of points that are in close proximity to other points.

    val_dict = dictionary_values(x_values, y_values)
    total_proximities = 0
    for x, y in zip(x_values, y_values):
        if surroundings_checker(val_dict, x, y, 3):
            total_proximities += 1
    print(f"Final Proximities: {total_proximities}")


def group_initialization(options):
    # Initialize a group with the given options.
    return {
        "all_options": [options],
        "last_x": options[0][0],
    }



def dict_pair_deletion(val_dict, x_coord, y_coord):
    # Remove a coordinate pair (x, y) from the value dictionary.
    del val_dict[x_coord][y_coord]
    if len(val_dict[x_coord]) == 0:
        del val_dict[x_coord]




def same_group_or_not(new_options, existing_groups):
    for group in existing_groups:
        next_x = group["last_x"] + spacing_of_option_box_height
        x_overlap, y_overlap = False, False
        
        # Checking if any option overlaps in the x-direction
        for option in new_options:
            if (option[0] > (next_x - 5)) and (option[0] < (next_x + 5)):
                x_overlap = True
                break
        if not x_overlap:
            continue

        # Checking if any option overlaps in the y-direction
        adjustment_needed = True
        for new_option in new_options:
            for prev_option in group["all_options"][-1]:
                if (new_option[1] < (prev_option[1] + 5)) and (new_option[1] > (prev_option[1] - 5)):
                    y_overlap = True
                    if new_options.index(new_option) == 0 and group["all_options"][-1].index(prev_option) == 0:
                        adjustment_needed = False
                    break
            if y_overlap:
                break
        if y_overlap:
            return True, group, adjustment_needed
    return False, None, None


def shift_options(options, count):
    new_options = options[:len(options) - count]
    for i in range(count):
        x = int(2 * new_options[0][0] - new_options[1][0])
        y = int(2 * new_options[0][1] - new_options[1][1])
        new_options = [(x, y, False)] + new_options
    return new_options


def group_adjustment(group, new_options):
    # Readjusting the group based on new options.
    prev_options = group["all_options"][-1]
    if prev_options[0][1] < new_options[0][1]:
        # Only need to adjust the current options
        count = 0
        last_group_option = group["all_options"][-1]
        for option in last_group_option:
            if (new_options[0][1] > (option[1] - 5)) and (new_options[0][1] < (option[1] + 5)):
                break
            count += 1
        shifted_options = shift_options(new_options, count)
        return group, new_options, shifted_options
    else:
        # Adjusting all previous options
        # Finding the count of shifts
        count = 0
        last_group_option = group["all_options"][-1][0]
        for option in new_options:
            if (last_group_option[1] > (option[1] - 5)) and (last_group_option[1] < (option[1] + 5)):
                break
            count += 1

        new_all_options = []
        for option_list in group["all_options"]:
            shifted_options = shift_options(option_list, count)
            new_all_options.append(shifted_options)
        group["all_options"] = new_all_options
        group["last_x"] = new_all_options[-1][0][0]
        return group, new_options, new_options


def group_addition(group, options):
    group["all_options"].append(options)
    group["last_x"] = options[0][0]


def question_grouping(question_beginnings):
    # Group questions based on their beginning coordinates.
    vertical_groups = []

    vertical_groups = []

    # Iterate over sorted x and y coordinates
    for x in sorted(question_beginnings.keys()):
        for y in sorted(question_beginnings[x].keys()):
            # Checking if the question belongs to an existing group
            belongs, group, need_adjustment = same_group_or_not(question_beginnings[x][y], vertical_groups)
            if belongs:
                if need_adjustment:
                    # If adjustment is needed, perform readjustment
                    group, temp1, new_options = group_adjustment(group, question_beginnings[x][y])
                    group_addition(group, new_options)
                else:
                    # Otherwise, add the question to the group
                    group_addition(group, question_beginnings[x][y])
            else:
                # If the question doesn't belong to any existing group, initializing a new group
                new_group = group_initialization(question_beginnings[x][y])
                vertical_groups.append(new_group)

    return vertical_groups

def are_group_combineable(group1, group2):
    first_options = group2["all_options"][0]
    last_options = group1["all_options"][-1]

    feasible_range = group1["last_x"] + 2.2 * spacing_of_option_box_height
    if feasible_range < first_options[0][0] or last_options[0][0] > first_options[0][0]:
        return False, 0, False

    # Defining the y coordinates of the first and last options in each group
    y11, y12, y21, y22 = first_options[0][1], first_options[-1][1], last_options[0][1], last_options[1][1]

    # Checking if there is overlap in the y direction between the two groups
    if not (y12 > y21 and y22 > y11):
        return False, 0, False

    # Checking if the first option of group2 aligns with the last option of group1
    if (y11 > (y21 - 5)) and (y11 < (y21 + 5)):
        return True, round((first_options[0][0] - last_options[0][0]) / spacing_of_option_box_height) - 1, False
    else:
        return True, round((first_options[0][0] - last_options[0][0]) / spacing_of_option_box_height) - 1, True



def group_merger(gp1, gp2, num_rows):
    img_rows = rows_of_image_builder(gp1["all_options"][-1], gp2["all_options"][0], num_rows)
    gp1["all_options"].extend(img_rows)
    gp1["all_options"].extend(gp2["all_options"])
    gp1["last_x"] = gp2["last_x"]


def rows_of_image_builder(opt1, opt2, num_rows):  # opt2 > opt1
    opts = []
    for i in range(num_rows):
        opt = []
        for j in range(len(opt1)):
            x = round(opt1[j][0] + (opt2[j][0] - opt1[j][0]) / (num_rows + 1))
            y = round(opt1[j][1] + (opt2[j][1] - opt1[j][1]) / (num_rows + 1))
            opt.append((x, y, False))
        opts.append(opt)
    return opts




def find_all_questions_beginings(x_vals, y_vals):
    # Finds all the beginnings of questions from non-max suppressed x and y values
    val_dict = dictionary_values(x_vals, y_vals)
    running_val_dict = copy.deepcopy(val_dict)
    total_remaining_points = len(x_vals)
    question_beginnings = {}  # x: {y: option_points}

    # Continue until all points are processed
    while total_remaining_points > 0:
        # Find the minimum x and y values
        min_x = min(running_val_dict.keys())
        all_points_min_x = horizontal_points(min_x, running_val_dict)
        min_y = min([pt[1] for pt in all_points_min_x])
        min_x = min([pt[0] for pt in all_points_min_x if pt[1] == min_y])

        # Check if it's the beginning of a question
        is_question, option_points = question_or_not(min_x, min_y, all_points_min_x)
        if is_question:
            if min_x not in question_beginnings:
                question_beginnings[min_x] = {}
            if min_y not in question_beginnings[min_x]:
                question_beginnings[min_x][min_y] = option_points

            # Remove all points belonging to the question
            for pt in option_points:
                if pt[2]:
                    total_remaining_points -= 1
                    dict_pair_deletion(running_val_dict, *pt[:2])
        else:
            total_remaining_points -= 1
            dict_pair_deletion(running_val_dict, min_x, min_y)
    return question_beginnings



def group_combiner(vertical_groups):
    combined_groups = []
    for i, group1 in enumerate(vertical_groups):
        for j, group2 in enumerate(vertical_groups):
            # Skiping if the same group or if either group is marked as invalid
            if (i == j) or "invalid" in group1 or "invalid" in group2:
                continue
            
            # Checking if the groups are combinable
            should_combine, mis_row, need_adj = are_group_combineable(group1, group2)
            if should_combine:
                # Marking the second group as invalid to prevent further processing
                group2["invalid"] = True
                # Adjusting the groups if needed before merging
                if need_adj:
                    if group1["all_options"][-1][0][1] > group2["all_options"][0][0][1]:
                        temp0, temp1, temp2 = group_adjustment(group1, group2["all_options"][0])
                    else:
                        temp0, temp1, temp2 = group_adjustment(group2, group1["all_options"][-1])
                # Merge the groups
                group_merger(group1, group2, mis_row)
    # Remove the invalid groups and return the combined groups
    return [group for group in vertical_groups if "invalid" not in group]



def option_kernel_selection():
    return np.full((box_height_of_option_choice, box_width_of_option_choice), -1)


def options_selected_sender(arr_of_image, option_list):
    kernel = option_kernel_selection()
    kernel_vals = []
    for option in option_list:
        kernel_vals.append(np.sum(arr_of_image[option[0]:option[0] + box_height_of_option_choice, option[1]:option[1] + box_width_of_option_choice] * kernel))
    return [val > threshold_selection for val in kernel_vals], kernel_vals


def ground_truth_reader(filename, skip_true_answer=False):
    # Reads ground truth data from a file.
    ground_truth = [[False for _ in range(5)] for _ in range(85)]
    custom_ans_indces = []

    # Read data from the file
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse each line of the file
    for i, line in enumerate(lines):
        parts = line.split()
        options = [False] * 5

        # Process the line if it contains answer information
        if len(parts) > 1:
            # Update the options list based on the provided answer
            for option in parts[1]:
                options[index_of_options[option]] = True

        # Check for custom answers if required
        if not skip_true_answer and len(parts) == 3 and parts[2] == "x":
            custom_ans_indces.append(i)

        # Update the ground truth data with the parsed information
        ground_truth[int(parts[0]) - 1] = options

    # Return the ground truth data and custom answer indices if needed
    if not skip_true_answer:
        return ground_truth, custom_ans_indces
    return ground_truth


def perform_selected_option_analytics(checked_options, ground_truths, question_markers):
    # Analyzes the selected options compared to the ground truth data.
    sel_min_max = [[50 * 50 * 255, -50 * 50 * 255] for _ in range(5)]
    un_sel_min_max = [[50 * 50 * 255, -50 * 50 * 255] for _ in range(5)]
    sel_min_max_idx = [[[0, 0], [0, 0]] for _ in range(5)]
    un_sel_min_max_idx = [[[0, 0], [0, 0]] for _ in range(5)]

    # Iterate over each question's selected options, ground truth data, and question markers
    for selected_options, ground_truth, question_marker in zip(checked_options, ground_truths, question_markers):
        # Compare each option for the current question
        for i, (option_value, selected, option_marker) in enumerate(zip(selected_options, ground_truth, question_marker)):
            # Update statistics for selected options
            if selected:
                if option_value < sel_min_max[i][0]:
                    sel_min_max[i][0] = option_value
                    sel_min_max_idx[i][0] = option_marker[:2]
                if option_value > sel_min_max[i][1]:
                    sel_min_max[i][1] = option_value
                    sel_min_max_idx[i][1] = option_marker[:2]
            # Update statistics for unselected options
            else:
                if option_value < un_sel_min_max[i][0]:
                    un_sel_min_max[i][0] = option_value
                    un_sel_min_max_idx[i][0] = option_marker[:2]
                if option_value > un_sel_min_max[i][1]:
                    un_sel_min_max[i][1] = option_value
                    un_sel_min_max_idx[i][1] = option_marker[:2]

    # Calculate additional statistics and return all results
    return sel_min_max, un_sel_min_max, \
           min(op[0] for op in sel_min_max), max(op[1] for op in sel_min_max), \
           min(op[0] for op in un_sel_min_max), max(op[1] for op in un_sel_min_max), \
           sel_min_max_idx, un_sel_min_max_idx


def create_next_row(row_1, row_2):
    next_row = []
    for o1, o2 in zip(row_1, row_2):
        next_row.append([o2[0] * 2 - o1[0], o2[1] * 2 - o1[1]])
    return next_row

def check_question_match_horizontally(options_0, options_1):
    y_change_intra = options_0[-1][1] - options_0[0][1]
    y_change_inter = options_1[0][1] - options_0[0][1]
    
    x_change_intra = options_0[-1][0] - options_0[0][0]
    x_change_inter = y_change_inter * x_change_intra / y_change_intra
    
    # Calculate expected x-coordinate of the first option in the second row
    x_val = options_0[0][0] + x_change_inter
    
    return abs(x_val - options_1[0][0]) < 5


def adding_begginning_ending_questions_if_needed(vertical_groups):
    if len(vertical_groups[0]["all_options"]) < 29:
        if check_question_match_horizontally(vertical_groups[0]["all_options"][0], vertical_groups[1]["all_options"][0]):
            if check_question_match_horizontally(vertical_groups[0]["all_options"][-1], vertical_groups[1]["all_options"][-1]):
                if len(vertical_groups[1]["all_options"]) < 29:
                    if check_question_match_horizontally(vertical_groups[0]["all_options"][0], vertical_groups[2]["all_options"][0]):
                        # Add rows at the top of both groups 0 and 1
                        vertical_groups[0]["all_options"] = [create_next_row(vertical_groups[0]["all_options"][1], vertical_groups[0]["all_options"][0])] + vertical_groups[0]["all_options"]
                        vertical_groups[1]["all_options"] = [create_next_row(vertical_groups[1]["all_options"][1], vertical_groups[1]["all_options"][0])] + vertical_groups[1]["all_options"]
                else:
                    # Add rows at the bottom of both groups 0 and 1
                    vertical_groups[0]["all_options"].append(create_next_row(vertical_groups[0]["all_options"][-2], vertical_groups[0]["all_options"][-1]))
                    vertical_groups[1]["all_options"].append(create_next_row(vertical_groups[1]["all_options"][-2], vertical_groups[1]["all_options"][-1]))
            else:
                # Add a row at the bottom of group 0
                vertical_groups[0]["all_options"].append(create_next_row(vertical_groups[0]["all_options"][-2], vertical_groups[0]["all_options"][-1]))
        else:
            # Add a row at the top of group 0
            vertical_groups[0]["all_options"] = [create_next_row(vertical_groups[0]["all_options"][1], vertical_groups[0]["all_options"][0])] + vertical_groups[0]["all_options"]
    
    # Check if the second group needs additional rows at the beginning or end
    if len(vertical_groups[1]["all_options"]) < 29:
        if check_question_match_horizontally(vertical_groups[0]["all_options"][0], vertical_groups[1]["all_options"][0]):
            if check_question_match_horizontally(vertical_groups[0]["all_options"][-1], vertical_groups[1]["all_options"][-1]):
                if len(vertical_groups[1]["all_options"]) < 29:
                    if check_question_match_horizontally(vertical_groups[0]["all_options"][0], vertical_groups[2]["all_options"][0]):
                        # Add rows at the top of both groups 0 and 1
                        vertical_groups[0]["all_options"] = [create_next_row(vertical_groups[0]["all_options"][1], vertical_groups[0]["all_options"][0])] + vertical_groups[0]["all_options"]
                        vertical_groups[1]["all_options"] = [create_next_row(vertical_groups[1]["all_options"][1], vertical_groups[1]["all_options"][0])] + vertical_groups[1]["all_options"]
                else:
                    # Add rows at the bottom of both groups 0 and 1
                    vertical_groups[0]["all_options"].append(create_next_row(vertical_groups[0]["all_options"][-2], vertical_groups[0]["all_options"][-1]))
                    vertical_groups[1]["all_options"].append(create_next_row(vertical_groups[1]["all_options"][-2], vertical_groups[1]["all_options"][-1]))
            else:
                # Add a row at the bottom of group 1
                vertical_groups[1]["all_options"].append(create_next_row(vertical_groups[1]["all_options"][-2], vertical_groups[1]["all_options"][-1]))
        else:
            # Add a row at the top of group 1
            vertical_groups[1]["all_options"] = [create_next_row(vertical_groups[1]["all_options"][1], vertical_groups[1]["all_options"][0])] + vertical_groups[1]["all_options"]
    
    # Check if the third group needs additional rows at the beginning or end
    if len(vertical_groups[2]["all_options"]) < 27:
        if check_question_match_horizontally(vertical_groups[2]["all_options"][0], vertical_groups[1]["all_options"][0]):
            # Add a row at the bottom of group 2
            vertical_groups[2]["all_options"].append(create_next_row(vertical_groups[2]["all_options"][-2], vertical_groups[2]["all_options"][-1]))
        else:
            # Add a row at the top of group 2
            vertical_groups[2]["all_options"] = [create_next_row(vertical_groups[2]["all_options"][1], vertical_groups[2]["all_options"][0])] + vertical_groups[2]["all_options"]
            # Add 1 at the top of 2


def selected_question_markers(image_array, question_markers):
    question_kernel_values = []
    selected_lists = []
    for question_marker in question_markers:
        selected_list, kernel_values = options_selected_sender(image_array, question_marker)
        question_kernel_values.append(kernel_values)
        selected_lists.append(selected_list)
    return question_kernel_values, selected_lists


def marker_array(groups):
    # Generates a marker array for questions.
    filtered_groups = [group for group in groups if len(group["all_options"]) >= 2]

    # Extract y values from filtered groups and sort them
    y_values = [int(group["all_options"][0][0][1]) for group in filtered_groups]
    sorted_y_values = sorted(y_values)

    # Add beginning and end questions if needed
    adding_begginning_ending_questions_if_needed([filtered_groups[y_values.index(y)] for y in sorted_y_values])

    # Generate the marker array for questions
    marker_array = []
    for y_value in sorted_y_values:
        index = y_values.index(y_value)
        selected_group = filtered_groups[index]
        for option_list in selected_group["all_options"]:
            option_markers = []
            for option in option_list:
                option_markers.append(option[:2])
            marker_array.append(option_markers)
    return marker_array
   


def kernel_handwritten():
    return np.ones((box_height_of_option_choice, text_width_custom))

def custom_answer_checker(question_marker, arr_of_image):
    image_patch = arr_of_image[question_marker[0][0]: question_marker[0][0] + box_height_of_option_choice,
                question_marker[0][1] - text_distance_custom_option: question_marker[0][1] + text_width_custom - text_distance_custom_option]
    kernel = kernel_handwritten()
    act_val = (image_patch * kernel).sum()
    return act_val <= ans_limit_custom, act_val

def custom_answer_getter(arr_of_image, q_markers, gd_truths):
    custom_ans_indces = []
    act_vals = []
    for i, gd_truth in enumerate(gd_truths):
        is_custom, act_val = custom_answer_checker(q_markers[i], arr_of_image)
        act_vals.append(act_val)
        if is_custom:
            custom_ans_indces.append(i)
    return custom_ans_indces, act_vals

def image_reader(image_path):
    img = Image.open(image_path).convert('L')
    if img.mode == "RGBA":
        return np.copy(np.asarray(img)[..., :3])
    else:
        return np.copy(np.asarray(img))

def groundtruth_to_text(gd_truths, all_custom_answers):
    map = {i:a for i, a in enumerate("ABCDE")}
    lines = []
    for i, gd_truth in enumerate(gd_truths):
        line = str(i+1) + " "
        for j, val in enumerate(gd_truth):
            if val:
                line += map[j]
        if i in all_custom_answers:
            line += " x"
        lines.append(line)
    return "\n".join(lines)

def file_writter(lines, filename):
    with open(filename, "w") as f:
        f.write(lines)

def image_grader(image_path, output_path):
    # Grades the image based on detected options and writes the result to an output file.
    arr_of_image = image_reader(img_path)
    kernel = kernel_option_finder()
    detected_img = bands_kernel(arr_of_image, kernel, False)

    indices = np.argpartition(detected_img.reshape(-1), -1000)[-1000:]
    indices = indices[np.argsort(-detected_img.reshape(-1)[indices])]

    x_vals, y_vals = get_x_y_from_flatten_arr(indices, detected_img.shape)
    x_vals, y_vals, rem_x, rem_y = top_k_not_close_to_each_other(x_vals, y_vals, 435)
    x_vals = [int(x) for x in x_vals]
    y_vals = [int(y) for y in y_vals]

    question_beginings = find_all_questions_beginings(x_vals, y_vals)

    all_vertical_groups = question_grouping(question_beginings)
    all_vertical_groups = group_combiner(all_vertical_groups)

    question_markers = marker_array(all_vertical_groups)
    all_kernel_vals, all_sel_vals = selected_question_markers(arr_of_image, question_markers)
    custom_ans_indices, act_vals = custom_answer_getter(arr_of_image, question_markers, all_sel_vals)

    lines = groundtruth_to_text(all_sel_vals, custom_ans_indices)
    file_writter(lines, output_path)

if __name__ == '__main__':    
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # initializing required parameters
    # Dimensions of the option box
    box_height_of_option_choice = 35
    box_width_of_option_choice = 33

    # Spacing between option boxes
    spacing_of_option_box_width = 60
    spacing_of_option_box_height = 48

    # Thresholds for question error checking
    threshold_width_avg_error_question = 3
    threshold_point_error_question = 2

    # Threshold for selecting options
    threshold_selection = -170000

    # Dimensions and limits for custom text (for handwritten answers)
    text_width_custom = 60
    ans_limit_custom = 514000
    text_distance_custom_option = 120

    # Mapping of option labels to indices
    index_of_options = {l: i for i, l in enumerate("ABCDE")}

    
    image_grader(img_path, output_path)