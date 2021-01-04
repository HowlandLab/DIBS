"""
Creating videos and everything related
"""

from typing import Any, List, Tuple, Union
from tqdm import tqdm
import cv2
import inspect
import multiprocessing
import numpy as np
import os
import random
# import sys
# import time
# import easygui  # docs: http://easygui.sourceforge.net/

from dibs import check_arg, config, statistics
from dibs.logging_dibs import get_current_function

logger = config.initialize_logger(__file__)

# HOW TO GET FPS: fps = video.get(cv2.cv.CV_CAP_PROP_FPS)


### In development
# Previuosly: fourcc='mp4v', but then videos weren't being created well
def make_labeled_video_according_to_frame(labels_list: Union[List, Tuple], frames_indices_list: Union[List, Tuple], output_file_name: str, video_source: str, current_behaviour_list: List[str] = (), output_fps=15, fourcc='avc1', output_dir=config.OUTPUT_PATH, text_colors_list: Union[Tuple, List] = (), **kwargs):
    """
    # PREVIOUSLY: fourcc='mp4v' was default
    # PREVIOUSLY code was 'H264'
    Make a video clip of an existing video

    :param labels_list: (List[Any]) a list of labels to be included onto the frames for the final video
    :param frames_indices_list: (List[int]) a list of frames by index to be labeled and included in final video
    :param video_source: (str) a path to a video file ___
    :param current_behaviour_list:
    :param output_file_name: (str)
    :param output_fps: (int)
    :param fourcc: (str)
    :param output_dir: (str)
    :param text_colors_list:

    :param kwargs:
        text_prefix : str

        font_scale : int

        text_colour_bgr : Tuple[int, int, int]

        rectangle_colour_bgr : Tuple[int, int, int]

        text_offset_x : int

        text_offset_y : int


    :return: None
    """
    font: int = cv2.FONT_HERSHEY_COMPLEX

    # Kwargs
    font_scale = kwargs.get('font_scale', config.DEFAULT_FONT_SCALE)
    rectangle_colour_bgr: Tuple[int, int, int] = kwargs.get('rectangle_bgr', config.DEFAULT_TEXT_BACKGROUND_BGR)  # 000=Black box?
    text_colour_bgr: Tuple[int, int, int] = kwargs.get('text_colour_bgr', config.DEFAULT_TEXT_BGR)
    text_prefix = kwargs.get('text_prefix', '')
    text_offset_x = kwargs.get('text_offset_x', 50)
    text_offset_y = kwargs.get('text_offset_y', 125)

    # Arg checking
    check_arg.ensure_is_file(video_source)
    check_arg.ensure_has_valid_chars_for_path(output_file_name)
    check_arg.ensure_is_dir(output_dir)
    check_arg.ensure_type(output_fps, int, float)  # TODO: uncomment this later
    # # Check kwargs
    check_arg.ensure_type(font_scale, int)
    check_arg.ensure_type(rectangle_colour_bgr, tuple)
    check_arg.ensure_type(rectangle_colour_bgr[0], int)
    check_arg.ensure_type(text_colour_bgr, tuple)
    check_arg.ensure_type(text_prefix, str)
    check_arg.ensure_type(text_offset_x, int)
    check_arg.ensure_type(text_offset_y, int)
    check_arg.ensure_type(text_colors_list, list, tuple)
    if len(labels_list) != len(frames_indices_list):
        non_matching_lengths_err = f'{get_current_function()}(): the number of labels and the list of frames do not match. Number of labels = {len(labels_list)} while number of frames = {len(frames_indices_list)}'
        logger.error(non_matching_lengths_err)
        raise ValueError(non_matching_lengths_err)
    ### Optional inputs
    # Text colors list
    if len(text_colors_list) != 0:
        if len(text_colors_list) != len(labels_list):
            err = f'The number of text colors list entries does not match the number of labels provided'
            logger.error(err)
            raise ValueError(err)
    else:
        text_colors_list = [text_colour_bgr for _ in range(len(labels_list))]
    # Behaviour text
    if len(current_behaviour_list) != 0:
        # First, arg check
        if len(current_behaviour_list) != len(labels_list):
            err = f"Incorrect # of behaviours doesn't match # of labels"  # TODO: low improve err msg
            logger.error(err)
            raise ValueError(err)

    else:
        current_behaviour_list = ['' for _ in range(len(labels_list))]

    ### Execute ###
    # Open source video
    cv2_source_video_object = cv2.VideoCapture(video_source)
    total_frames_of_source_vid = int(cv2_source_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"Total # of frames in video: {total_frames_of_source_vid}")
    logger.debug(f"Is it opened? {cv2_source_video_object.isOpened()}")

    # Get dimensions of first frame in video, use that to instantiate VideoWriter parameters
    cv2_source_video_object.set(1, 0)
    is_frame_retrieved, frame = cv2_source_video_object.read()
    if not is_frame_retrieved:
        err = f'Frame not retrieved. Is video path correct? video_source = {video_source} '
        logger.error(err)
        raise ValueError(err)
    height, width, _layers = frame.shape

    # Open video writer object
    four_character_code = cv2.VideoWriter_fourcc(*fourcc)
    full_output_video_path = os.path.join(output_dir, f'{output_file_name}.mp4')
    logger.debug(f'Video saving to: {full_output_video_path}')
    video_writer = cv2.VideoWriter(
        full_output_video_path,                                 # Full output file path
        four_character_code,                                    # fourcc -- four character code
        output_fps,                                             # fps
        (width, height)                                         # frameSize
    )

    # Loop over all requested frames, add text, then append to list for later video creation
    for i in range(len(frames_indices_list)):
        label, frame_idx, text_color_tuple = labels_list[i], frames_indices_list[i], text_colors_list[i]
        current_behaviour = current_behaviour_list[i]
        # logger.debug(f'label, frame_idx = {label}, {frame_idx} // type(label), type(frame_idx) = {type(label)}, {type(frame_idx)}')  # TODO: remove this line after type debugging
        cv2_source_video_object.set(1, frame_idx)
        is_frame_retrieved, frame = cv2_source_video_object.read()
        if not is_frame_retrieved:
            no_frame_err = f'frame index ({frame_idx}) not found not found. Total frames in ' \
                           f'video: {int(cv2_source_video_object.get(cv2.CAP_PROP_FRAME_COUNT))}'
            raise Exception(no_frame_err)

        text_for_frame = f'Frame index: {frame_idx} // {text_prefix}Current Assignment: {label} // Current behaviour label: {current_behaviour}'

        text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]

        # # Old, working implementation below
        # # Add background colour for text
        # box_top_left, box_top_right = (
        #     (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
        #     (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
        # )
        # cv2.rectangle(frame, box_top_left, box_top_right, rectangle_colour_bgr, cv2.FILLED)
        # # Add text
        # cv2.putText(
        #     img=frame,
        #     text=str(text_for_frame),
        #     org=(text_offset_x, text_offset_y),
        #     fontFace=font,
        #     fontScale=font_scale,
        #     color=text_color_tuple,
        #     thickness=1
        # )
        # # numpy_frames.append(frame)


        # # New attempt implementation for functional addition of text/color
        # # def put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, text_color_tuple: Tuple[int], rectangle_colour_bgr: Tuple[int], disposition_x: int = 0, disposition_y: int = 0):
        # # 1/2: top level text
        frame = put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, text_color_tuple, rectangle_colour_bgr)
        frame = add_border(frame, color=text_colour_bgr, pixel_width=10)

        # 2/2: Bottom level text (STILL WIP! -- put_text_over_box_on_image() needs to be debugged first before uncomment below
        # frame = put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, text_color_tuple, rectangle_colour_bgr, disposition_x=100, disposition_y=50)

        # Write to video
        video_writer.write(frame)

    ###########################################################################################
    ### Now that all necessary frames are extracted & labeled, create video with them.
    # Extract first image in images list. Get dimensions for video.
    # All done. Release video, clean up, then return.
    video_writer.release()
    cv2.destroyAllWindows()
    logger.debug(f'{get_current_function()}(): Done writing video.')
    return


def put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, text_color_tuple: Tuple[int, int, int], rectangle_colour_bgr: Tuple[int, int, int], disposition_x: int = 0, disposition_y: int = 0) -> np.ndarray:
    text_offset_x = text_offset_x + disposition_x
    text_offset_y = text_offset_y + disposition_y
    text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]

    box_top_left: tuple = (text_offset_x - 12, text_offset_y + 12)
    box_top_right: tuple = (text_offset_x + text_width + 12, text_offset_y - text_height - 8)

    # Add background rectangle for text contrast
    cv2.rectangle(frame, box_top_left, box_top_right, rectangle_colour_bgr, cv2.FILLED)

    # Add text
    cv2.putText(
        img=frame,
        text=str(text_for_frame),
        org=(text_offset_x, text_offset_y),
        fontFace=font,
        fontScale=font_scale,
        color=text_color_tuple,
        thickness=1
    )
    return frame


def add_border(frame: np.ndarray, color: Tuple[int, int, int], pixel_width: int):
    """
    Add color border around frame without changing dimensions
    :param frame:
    :param color:
    :param pixel_width:
    :return:
    """
    # Set number of pixels of insets on respective side
    top, bottom, left, right = pixel_width, pixel_width, pixel_width, pixel_width

    image = cv2.copyMakeBorder(frame[top:-bottom, left:-right, :], top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=color)

    return image


### Previously used

def generate_video_with_labels(labels: Union[List, Tuple, np.ndarray], source_video_file_path, output_file_name, output_fps, fourcc='mp4v', output_dir_path=config.OUTPUT_PATH, **kwargs):
    """
    Given a video and labels for behaviours that correspond to that given video, create a new video
    which has text labels on it.
    :param labels: (list, tuple, collection of strings)
    :param source_video_file_path: (str) the absolute path to the video which will be labeled. This video
        will not be overwritten.
    :param output_file_name: (str) the name of the file when it is done being created. Do not
        include the path or extension, just the name.
    :param output_fps:
    :param fourcc:
    :param output_dir_path:
    :param kwargs:
        font_scale (int):
        rectangle_br: (tuple(int, int, int)):  Sets colour for rectangle around text
    :return:
    """

    text_color_bgr = (255, 255, 255)  # 255*3 == white?

    # Get kwargs, check args
    if not os.path.isfile(source_video_file_path):
        not_a_vid_err = f'This is not a file/video: {source_video_file_path} / todo: elaborate'
        logger.error(not_a_vid_err)
        raise ValueError(not_a_vid_err)
    if not isinstance(labels, list) and not isinstance(labels, tuple) and not isinstance(labels, np.ndarray):
        type_err = f'`labels` had an unexpected type: {type(labels)}'  # TODO: flesh out err msg
        logger.error(type_err)
        raise TypeError(type_err)
    font_scale = kwargs.get('font_scale', 1)
    rectangle_bgr = kwargs.get('rectangle_bgr', (0, 0, 0))  # 000=Black box?
    check_arg.ensure_type(output_fps, int)
    check_arg.ensure_type(fourcc, str)
    check_arg.ensure_type(font_scale, int)
    check_arg.ensure_type(rectangle_bgr, tuple)

    ### Do main work
    # # TODO: rework this comment: 1/2: Iterate over video frames, add text labels, add frame to frames_list
    font = cv2.FONT_HERSHEY_COMPLEX
    four_character_code = cv2.VideoWriter_fourcc(*fourcc)

    cv2_video_object = cv2.VideoCapture(source_video_file_path)
    total_frames_of_source_vid = int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f'Total frames: {total_frames_of_source_vid}')

    is_frame_retrieved, frame = cv2_video_object.read()
    height, width, _layers = frame.shape

    # Open video writer object
    video_writer = cv2.VideoWriter(
        os.path.join(output_dir_path, f'{output_file_name}.mp4'),   # Full output file path
        four_character_code,                                        # fourcc
        output_fps,                                                 # fps
        (width, height)                                             # frameSize
    )

    # Loop over frames, assign labels to all
    print('Is it opened?', cv2_video_object.isOpened())
    done = False
    i, frame_count = 0, 0

    while cv2_video_object.isOpened() and not done:

        is_frame_retrieved, frame = cv2_video_object.read()
        if not is_frame_retrieved:
            logger.debug(f'Is frame retrieved? {is_frame_retrieved}')
            done = True
            cv2_video_object.release()
            break
        text_for_frame = labels[i]

        ### TIGHTLY BOUND - TEXT ADDITION ###
        # TODO: Note: if the label is not a string, the below line will fail!
        text_width, text_height = cv2.getTextSize(str(text_for_frame), font, fontScale=font_scale, thickness=1)[0]

        text_offset_x, text_offset_y = 50, 50  # TODO: low: address magic variables later
        box_coordinates = (
            (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
            (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
        )
        cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img=frame, text=labels[i], org=(text_offset_x, text_offset_y),
                    fontFace=font, fontScale=font_scale, color=text_color_bgr, thickness=1)

        ###

        # Wrap up
        #frames_list.append(frame)
        video_writer.write(frame)

        i += 1
        frame_count += 1

        cv2_video_object.set(1, frame_count)
        if i % 100 == 0:
            logger.debug(f'Done iter {i}')
    logger.debug(f'Is it opened? {cv2_video_object.isOpened()}')

    ###########################################################################################
    # 2/2: Write video with object
    # Extract first image in images list. Set dimensions.


    # Loop over all images and write to file (as video) with video writer

    # All done. Release video, clean up, and return.
    video_writer.release()
    cv2.destroyAllWindows()

    return


def generate_frame_filename(frame_idx: int, ext=config.FRAMES_OUTPUT_FORMAT) -> str:
    """ Create a standardized way of naming frames from read-in videos """
    # TODO: low: move this func. Writing to file likely won't happen much in future, but do not deprecate this.
    total_num_length = 6
    leading_zeroes = max(total_num_length - len(str(frame_idx)), 0)
    name = f'frame_{"0"*leading_zeroes}{frame_idx}.{ext}'
    return name


def write_video_with_existing_frames(video_path, frames_dir_path, output_vid_name, output_fps=config.OUTPUT_VIDEO_FPS):  # TODO: <---------------------------------- Used just fine --------------------------------------
    """

    :param video_path:
    :param frames_dir_path:
    :param output_vid_name:
    :param output_fps:
    :return:
    """
    # TODO: add option to change output format (something other than mp4
    # Get (all) existing frames to be written
    frames = [x for x in os.listdir(config.FRAMES_OUTPUT_PATH) if x.endswith(f'.{config.FRAMES_OUTPUT_FORMAT}')]

    # Extract first image in images list. Set dimensions.
    four_character_code = cv2.VideoWriter_fourcc(*'mp4v')  # TODO: ensure fourcc can be change-able
    first_image = cv2.imread(os.path.join(frames_dir_path, frames[0]))
    height, width, _layers = first_image.shape

    # Loop over the range generated from the total unique labels available

    # Get video object, prep variables
    cv2_video_object: cv2.VideoCapture = cv2.VideoCapture(video_path)
    total_frames = int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f'Total frames: {total_frames}')

    # Open video writer object
    video_writer = cv2.VideoWriter(
        os.path.join(config.VIDEO_OUTPUT_FOLDER_PATH, f'{output_vid_name}.mp4'),  # filename
        four_character_code,  # fourcc
        output_fps,  # fps
        (width, height)  # frameSize
    )

    # Loop over all images and write to file with video writer
    log_every, i = 0, 250
    for image in tqdm(frames, desc='Writing video...', disable=True if config.stdout_log_level=='DEBUG' else False):  # TODO: low: add progress bar
        video_writer.write(cv2.imread(os.path.join(frames_dir_path, image)))
        # if i % log_every == 0:
        #     logger.debug(f'Working on iter: {i}')
        # i += 1
    video_writer.release()
    cv2.destroyAllWindows()
    return


def write_annotated_frames_to_disk_from_video_NEW(path_to_video: str, labels, fps: int = config.VIDEO_FPS, output_path: str = config.FRAMES_OUTPUT_PATH, pct_frames_to_label: float = config.PERCENT_FRAMES_TO_LABEL):
    """ * LEGACY *
    This function serves to supersede the old 'vid2frame()' function for future clarity.

    Extracts frames every 100ms to match the labels for visualizations  # TODO: Q: are we sure it pulls frames every 100ms when the FPS is variable?

    Assumptions:
        -

    :param path_to_video: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    :param pct_frames_to_label:
    # TODO: med: analyze use of magic variables in func.
    """
    if not os.path.isfile(path_to_video):  # Check if path to video exists.
        err = f'{__name__}:{inspect.stack()[0][3]}Path to video was not found. Path = {path_to_video}'
        logger.error(err)
        raise ValueError(err)

    #
    frames_to_skip_after_each_write = round(1 / pct_frames_to_label)
    cv2_video_object = cv2.VideoCapture(path_to_video)
    progress_bar = tqdm(total=int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Writing images to file...')
    # labels = np.hstack((labels[0], labels))  # fill the first frame  # TODO: why need to fill first frame?
    frame_count = 0
    i = 0
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    while cv2_video_object.isOpened():
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            # Prepare writing info onto image
            text_width, text_height = cv2.getTextSize(labels[i], font, fontScale=font_scale, thickness=1)[0]
            # TODO: evaluate magic variables RE: text offsetting on images
            text_offset_x, text_offset_y = 50, 50
            box_coordinates = (
                (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
                (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
            )
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img=frame, text=labels[i], org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
            # Write to image
            image_name = generate_frame_filename(frame_count)
            cv2.imwrite(os.path.join(output_path, image_name), frame)

            # End of loop steps. Save & set metrics, prepare for next frame, and update progress bar.
            frame_count += frames_to_skip_after_each_write
            cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
            progress_bar.update(frames_to_skip_after_each_write)
            i += 1
        else:  # No more frames left to retrieve. Release object and finish.
            cv2_video_object.release()
            break
    progress_bar.close()
    return


### Frame writing (new?)

def write_individual_frame_to_file(is_frame_retrieved: bool, frame: np.ndarray, label, frame_idx, output_path=config.FRAMES_OUTPUT_PATH):
    """ * NEW *
    (For use in multiprocessing frame-writing.
    :param is_frame_retrieved:
    :param frame:
    :param label:
    :param frame_idx:
    :param output_path:
    :return:
    """
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr_black = (0, 0, 0)
    color_white_bgr = (255, 255, 255)
    if is_frame_retrieved:
        # Prepare writing info onto image
        text_for_frame = f'Label__'
        # Try appending label
        # TODO: OVERHAUL LABELEING
        try:
            label_word = config.map_group_to_behaviour[label]
            text_for_frame += label_word
        except KeyError:
            text_for_frame += f'NotFound. Group: {label}'
            label_not_found_err = f'Label number not found: {label}. '
            logger.error(label_not_found_err)
        except IndexError as ie:
            index_err = f'Index error. Could not index i ({frame_idx}) onto labels. / ' \
                        f'is_frame_retrieved = {is_frame_retrieved} / ' \
                        f'Original exception: {repr(ie)}'
            logger.error(index_err)
            raise IndexError(index_err)
        else:
            text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]
            # TODO: evaluate magic variables RE: text offsetting on images
            text_offset_x, text_offset_y = 50, 50
            box_coordinates_topleft, box_coordinates_bottom_right = (
                (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
                (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
            )
            cv2.rectangle(frame, box_coordinates_topleft, box_coordinates_bottom_right, rectangle_bgr_black, cv2.FILLED)
            cv2.putText(img=frame, text=text_for_frame, org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=color_white_bgr, thickness=1)
            # Write to image

            image_name = generate_frame_filename(frame_idx)
            cv2.imwrite(os.path.join(output_path, image_name), frame)
    return 1


def write_annotated_frames_to_disk_from_video_source_with_multiprocessing(path_to_video: str, labels, pct_frames_to_label: float = config.PERCENT_FRAMES_TO_LABEL) -> None:
    """
    A new multiprocessed version of writing frames to disk (using legacy execution flow) to
    leverage multiprocessing (optional with N_JOBS I suppose) just because original implementation is so slow.
    """
    # TODO: resolve fails

    # Arg checking
    assert os.path.isfile(path_to_video), f'Video does not exist: {path_to_video}'
    # Do
    frames_to_skip_after_each_write = round(1 / pct_frames_to_label)
    cv2_video_object = cv2.VideoCapture(path_to_video)
    total_frames = int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f'Total frames: {total_frames}')  # Debugging
    logger.debug(f'Initial labels shape: {labels.shape}')  # Debugging

    labels = np.hstack((labels[0], labels))  # fill the first frame  # TODO: why need to fill first frame?
    logger.debug(f'Labels shape after padding: {labels.shape}')

    i_frame, frame_count = 0, 0

    frames_queue: List[Tuple[bool, Any, str, int]] = []
    # queue up images to write
    done = False
    # Iterate over frames extracted from video. Generate a queue of frames to be labeled later.
    while not done:
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            frames_queue.append((is_frame_retrieved, frame, labels[i_frame], frame_count))
            # Save & set metrics, prepare for next frame, and update progress bar
            frame_count += frames_to_skip_after_each_write
            # Skip ahead to a specified frame?
            cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
            # progress_bar.update(frames_to_skip_after_each_write)
            i_frame += 1
        else:  # No more frames left to retrieve. Release object and finish.
            done = True
            cv2_video_object.release()
            break

    # Utilize multiprocessing to label those frames in the queue
    with multiprocessing.Pool(config.N_JOBS) as pool:
        # Set up function to be executed asynchronously
        results = [pool.apply_async(
            write_individual_frame_to_file,
            (is_frame_retrieved, frame, label, i, frames_to_skip_after_each_write), )
            for is_frame_retrieved, frame, label, i in frames_queue]
        # Execute
        results = [f.get() for f in results]

    cv2_video_object.release()

    return


### Video creation - Legacy ############################################################################################

@config.log_function_entry_exit(logger)
def write_annotated_frames_to_disk_from_video_LEGACY(path_to_video: str, labels, fps: int = config.VIDEO_FPS, output_path: str = config.FRAMES_OUTPUT_PATH, pct_frames_to_label: float = config.PERCENT_FRAMES_TO_LABEL):
    """ * LEGACY *
    This function serves to supersede the old 'vid2frame()' function for future clarity.

    Extracts frames every 100ms to match the labels for visualizations  # TODO: Q: are we sure it pulls frames every 100ms when the FPS is variable?

    Assumptions:
        -

    :param path_to_video: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    :param pct_frames_to_label:
    # TODO: med: analyze use of magic variables in func.
    """
    if not os.path.isfile(path_to_video):  # Check if path to video exists.
        err = f'{__name__}:{inspect.stack()[0][3]}Path to video was not found. Path = {path_to_video}'
        logger.error(err)
        raise ValueError(err)

    #
    frames_to_skip_after_each_write = round(1 / pct_frames_to_label)  # frames_to_skip_after_each_write = round(fps * (1 / pct_frames_to_label))  # TODO: implement properly
    set_unique_labels = set(np.unique(labels))
    cv2_video_object = cv2.VideoCapture(path_to_video)
    progress_bar = tqdm(total=int(cv2_video_object.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Writing images to file...')
    # width, height = cv2_video_object.get(3), cv2_video_object.get(4)  # TODO: low: address unused variables
    labels = np.hstack((labels[0], labels))  # fill the first frame  # TODO: why need to fill first frame?
    frame_count = 0
    i = 0
    font_scale, font = 1, cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)

    while cv2_video_object.isOpened():
        is_frame_retrieved, frame = cv2_video_object.read()
        if is_frame_retrieved:
            # Prepare writing info onto image
            text_for_frame = f'Label__'
            # TODO: OVERHAUL LABELEING
            # Try appending label
            try:
                label_word = config.map_group_to_behaviour[labels[i]]
                text_for_frame += label_word
            except KeyError:
                text_for_frame += f'NotFound. Group: {labels[i]}'
                label_not_found_err = f'Label number not found: {labels[i]}. Set of unique ' \
                                      f'label numbers = {set_unique_labels}'
                logger.error(label_not_found_err)
            except IndexError as ie:
                index_err = f'Index error. Could not index i ({i}) onto labels (labels.shape={labels.shape}) / ' \
                            f'is_frame_retrieved = {is_frame_retrieved} / ' \
                            f'Original exception: {repr(ie)}'
                logger.error(index_err)
                raise IndexError(index_err)

            text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]
            # TODO: evaluate magic variables RE: text offsetting on images
            text_offset_x, text_offset_y = 50, 50
            box_coordinates = (
                (text_offset_x - 12, text_offset_y + 12),  # pt1, or top left point
                (text_offset_x + text_width + 12, text_offset_y - text_height - 8),  # pt2, or bottom right point
            )
            cv2.rectangle(frame, box_coordinates[0], box_coordinates[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img=frame, text=text_for_frame, org=(text_offset_x, text_offset_y),
                        fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
            # Write to image
            image_name = f'frame_{frame_count+1}.png'
            cv2.imwrite(os.path.join(output_path, image_name), frame)

            # Save & set metrics, prepare for next frame, and update progress bar
            frame_count += frames_to_skip_after_each_write
            cv2_video_object.set(1, frame_count)  # first arg: 'propID' (like property ID), second arg is 'value'
            progress_bar.update(frames_to_skip_after_each_write)
            i += 1
        else:  # No more frames left to retrieve. Release object and finish.
            cv2_video_object.release()
            break
    progress_bar.close()
    return


def create_labeled_example_videos_by_label(labels, critical_behaviour_minimum_duration=3, num_randomly_generated_examples=5, frame_dir=config.FRAMES_OUTPUT_PATH, output_path=config.VIDEO_OUTPUT_FOLDER_PATH, output_fps=5, fourcc_extension='mp4v') -> None:
    """ * LEGACY *
    (Generalized create_labeled_video() function that works between _py, _umap, and _voc submodules)
    TODO: low: Describe
    :param labels: 1D array, labels from training or testing
    :param critical_behaviour_minimum_duration: scalar, minimum duration for random selection of behaviors, default 300ms
    :param num_randomly_generated_examples: scalar, number of randomly generated examples, default 5  # TODO: low: default is actually 3..does counts refer to cv2.VideoWriter pathing?
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    :param output_fps:
    """
    # Create list of only .png images found in the frames directory
    images: List[str] = [img for img in os.listdir(frame_dir) if img.endswith(f".{config.FRAMES_OUTPUT_FORMAT}")]
    if len(images) <= 0:
        empty_list_err = f'{inspect.stack()[0][3]}(): Zero .png frames were found in {frame_dir}. ' \
                         f'Could not created labeled video. Exiting early.'  # TODO:
        logger.error(empty_list_err)
        # raise ValueError(empty_list_err)
        return
    statistics.sort_list_nicely_in_place(images)

    # Extract first image in images list. Set dimensions.
    four_character_code = cv2.VideoWriter_fourcc(*fourcc_extension)
    first_image = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _layers = first_image.shape

    # ?  TODO
    labels_list, idx_list, lengths_list = statistics.augmented_runlength_encoding(labels)
    ranges_list, idx2_list = [], []

    # Loop over lengths
    for idx_length_i, length_i in enumerate(lengths_list):
        if length_i >= critical_behaviour_minimum_duration:
            ranges_list.append(range(idx_list[idx_length_i], idx_list[idx_length_i]+length_i))
            idx2_list.append(idx_length_i)

    # Loop over the range generated from the total unique labels available
    for idx_label in tqdm(range(len(np.unique(labels))), desc=f'{inspect.stack()[0][3]}(): Creating video (TODO: update this bar description)'):
        a: List[range] = []  # TODO: low: `a` needs more description
        for idx_range in range(len(ranges_list)):  # TODO: low: could making this loop into a comprehension be more readable?
            if labels_list[idx2_list[idx_range]] == idx_label:
                a += [ranges_list[idx_range], ]  # Previously: a.append(ranges[idx_range])
        try:
            random_ranges = random.sample(a, min(len(a), num_randomly_generated_examples))
            for idx_random_range, random_range_i in enumerate(random_ranges):
                grp_images = []
                # Loop over list of randomly generated ranges
                for a_random_range in random_ranges[idx_random_range]:  # TODO: substitute for `random_range_i`?
                    # Aggregate images into a list that correspond to the randomly generated numbers/ranges
                    try:
                        images_of_interest = images[a_random_range]
                    except IndexError as ie:  # Index out of range
                        # Original implementation allowed this part to fail silently...address this later when there is time!
                        index_err = f'{inspect.stack()[0][3]}(): Indexing error. ' \
                                    f'The "random range" appears to be out of bounds for the ' \
                                    f'images actually available. Check logic above. ' \
                                    f'Originally, this error failed silently but it should be addressed later. ' \
                                    f'Original error: {repr(ie)}'
                        logger.error(index_err)
                    else:  # Execute else block if no errors
                        grp_images.append(images_of_interest)

                if len(grp_images) > 0:
                    video_name = f'group_{idx_label} / example_{idx_random_range}.mp4'
                    # Open video writer object
                    video_writer = cv2.VideoWriter(
                        os.path.join(output_path, video_name),
                        four_character_code,
                        output_fps,
                        (width, height)
                    )
                    # Loop over all images and write to file with video writer
                    for image in grp_images:
                        video_writer.write(cv2.imread(os.path.join(frame_dir, image)))
                    # Release video writer and continue

                    video_writer.release()

                cv2.destroyAllWindows()

        except Exception as e:  # TODO: low: exception is very general. Address?
            err = f'{inspect.stack()[0][3]}: Unexpected exception occurred when trying to make video. ' \
                  f'Review exception and address/make function more robust instead of failing silently. ' \
                  f'Exception is: {repr(e)}.'
            logger.error(err)

            raise e
    return

