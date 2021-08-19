"""
Creating videos and everything related
"""

from typing import List, Tuple, Union
from tqdm import tqdm
import cv2
import numpy as np
import os
# import easygui  # docs: http://easygui.sourceforge.net/

from dibs import check_arg, config, statistics
from dibs.logging_enhanced import get_current_function

logger = config.initialize_logger(__name__)

# HOW TO GET FPS: fps = video.get(cv2.cv.CV_CAP_PROP_FPS)

def make_video_from_multiple_sources(
        data_source_with_video_clip_tuples, # has clips in order to be read
        data_source_to_video_path, # for opening video files to read
        output_file_name,
        output_dir,
        text_prefix,
        output_fps=15,
        fourcc='mp4v',
        **kwargs):
    """ NOTE: Mostly copied from "make_labeled_video_according_to_frame"
    Make a video clip, from all input videos in data_source_to_video_path based on the
    clips in data_source_with_video_clip_tuples.

    # PREVIOUSLY: fourcc='mp4v' was default
    # PREVIOUSLY code was 'H264'

    :param data_source_with_video_clip_tuples:
    :param data_source_to_video_path:
    :param output_file_name: (str)
    :param output_dir:
    :param text_prefix:
    :param output_fps: (int)
    :param fourcc: (str)

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
    # text_prefix = kwargs.get('text_prefix', '')
    text_offset_x = kwargs.get('text_offset_x', 50)
    text_offset_y = kwargs.get('text_offset_y', 125)

    # Arg checking
    # check_arg.ensure_is_file(video_file_path)
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

    ### Execute ###
    # Open source video

    def _open_video_file(video_path):
        # AARONT: TODO: Will this fail if the video doesn't exist?
        cv2_source_video_object = cv2.VideoCapture(video_path)
        total_frames_of_source_vid = int(cv2_source_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"Total # of frames in video: {total_frames_of_source_vid}")
        logger.debug(f"Is it opened? {cv2_source_video_object.isOpened()}")
        return cv2_source_video_object


    data_source_to_open_video_file = {
        data_source: _open_video_file(video_path)
        for data_source, video_path in data_source_to_video_path.items()
    }

    # Get dimensions of first frame in video, use that to instantiate VideoWriter parameters
    temp_data_source = list(data_source_to_open_video_file.keys())[0]
    cv2_source_video_object = data_source_to_open_video_file[temp_data_source]
    cv2_source_video_object.set(1, 0)
    is_frame_retrieved, frame = cv2_source_video_object.read()
    if not is_frame_retrieved:
        err = f'Frame not retrieved. Is video path correct? data source associated with video: {temp_data_source}'
        logger.error(err)
        raise ValueError(err)
    height, width, _layers = frame.shape

    # Open video writer object
    four_character_code = cv2.VideoWriter_fourcc(*fourcc)
    full_output_video_path = os.path.join(output_dir, f'{output_file_name}.mp4') # AARONT: TODO: Why is this mp4 and fourcc is avc1?
    logger.debug(f'Video saving to: {full_output_video_path}')

    # AARONT: TODO: Allow argument for video writer to be passed in, then we can call this function repeatedly and pass a vid writer through.
    video_writer = cv2.VideoWriter(
        full_output_video_path,                                 # Full output file path
        four_character_code,                                    # fourcc -- four character code
        output_fps,                                             # fps
        (width, height)                                         # frameSize
    )

    # Return list of tuples representing a clip from a video, all from the same source.
    # list[tuple(label, user_assignment_label, color, frame_idx), ...]
    # Loop over all requested frames, add text, then append to list for later video creation
    for data_source, clip in data_source_with_video_clip_tuples:
        cv2_source_video_object = data_source_to_open_video_file[data_source]
        for label, current_behaviour, color, frame_idx in clip:
            # current_behaviour = current_behaviour_list[i] # AARONT: TODO: Do we need both label and assignment? Yes
            # logger.debug(f'label, frame_idx = {label}, {frame_idx} // type(label), type(frame_idx) = {type(label)}, {type(frame_idx)}')  # TODO: remove this line after type debugging
            cv2_source_video_object.set(1, frame_idx)
            is_frame_retrieved, frame = cv2_source_video_object.read()
            if not is_frame_retrieved:
                no_frame_err = f'frame index ({frame_idx}) not found not found. Total frames in ' \
                               f'video: {int(cv2_source_video_object.get(cv2.CAP_PROP_FRAME_COUNT))}'
                raise Exception(no_frame_err)

            text_for_frame = f'Frame index: {frame_idx} // {text_prefix}Current Assignment: {label} // Current behaviour label: {current_behaviour}'

            text_width, text_height = cv2.getTextSize(text_for_frame, font, fontScale=font_scale, thickness=1)[0]

            # # New attempt implementation for functional addition of text/color
            # # def put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, text_color_tuple: Tuple[int], rectangle_colour_bgr: Tuple[int], disposition_x: int = 0, disposition_y: int = 0):
            # # 1/2: top level text
            frame = put_text_over_box_on_image(frame, text_for_frame, font, font_scale, text_offset_x, text_offset_y, color, rectangle_colour_bgr)
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

### In development
# Previuosly: fourcc='mp4v', but then videos weren't being created well
def make_labeled_video_according_to_frame(labels_list: Union[List, Tuple], frames_indices_list: Union[List, Tuple],
                                          output_file_name: str, video_file_path: str,
                                          current_behaviour_list: List[str] = (), output_fps=15, fourcc='avc1',
                                          output_dir=config.OUTPUT_PATH, text_colors_list: Union[Tuple, List] = (),
                                          **kwargs):
    """
    # PREVIOUSLY: fourcc='mp4v' was default
    # PREVIOUSLY code was 'H264'
    Make a video clip of an existing video

    :param labels_list: (List[Any]) a list of labels to be included onto the frames for the final video
    :param frames_indices_list: (List[int]) a list of frames by index to be labeled and included in final video
    :param video_file_path: (str) a path to a video file ___
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
    check_arg.ensure_is_file(video_file_path)
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
    cv2_source_video_object = cv2.VideoCapture(video_file_path)
    total_frames_of_source_vid = int(cv2_source_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.debug(f"Total # of frames in video: {total_frames_of_source_vid}")
    logger.debug(f"Is it opened? {cv2_source_video_object.isOpened()}")

    # Get dimensions of first frame in video, use that to instantiate VideoWriter parameters
    cv2_source_video_object.set(1, 0)
    is_frame_retrieved, frame = cv2_source_video_object.read()
    if not is_frame_retrieved:
        err = f'Frame not retrieved. Is video path correct? video_source = {video_file_path} '
        logger.error(err)
        raise ValueError(err)
    height, width, _layers = frame.shape

    # Open video writer object
    four_character_code = cv2.VideoWriter_fourcc(*fourcc)
    full_output_video_path = os.path.join(output_dir, f'{output_file_name}.mp4')
    logger.debug(f'Video saving to: {full_output_video_path}')

    # AARONT: TODO: Allow argument for video writer to be passed in, then we can call this function repeatedly and pass a vid writer through.
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

def generate_frame_filename(frame_idx: int, ext=config.FRAMES_OUTPUT_FORMAT) -> str:
    """ Create a standardized way of naming frames from read-in videos """
    # TODO: low: move this func. Writing to file likely won't happen much in future, but do not deprecate this.
    total_num_length = 6
    leading_zeroes = max(total_num_length - len(str(frame_idx)), 0)
    name = f'frame_{"0"*leading_zeroes}{frame_idx}.{ext}'
    return name


def write_video_with_existing_frames(video_path, frames_dir_path, output_vid_name, output_fps=config.OUTPUT_VIDEO_FPS):  # TODO: <---------------------------------- Used just fine --------------------------------------
    """
    TODO: Purpose seems to have been making video creation faster and/or allowing different formats to be created easily.
          AND for use in multi-processing frame writing?
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


