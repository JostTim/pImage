# -*- coding: utf-8 -*-

"""Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Tue Oct 12 17:34:44 2021
@author: Timothe
</div>
"""

from multiprocessing import Manager  # , Pool, TimeoutError
import sys, time
# from .video.readers import VideoReader
# from .video.writers import VideoWriter


class StandardConverter:
    def __init__(self, input_path, output_path, **kwargs):
        m = Manager()
        self.read_queue = m.Queue(30)  # max 30 frames in ram at the same time if reading
        # is faster than writing (it is)
        # self.transformed_queue = m.Queue()
        self.message_queue = m.Queue()

        self.input_path = input_path
        self.output_path = output_path
        self.reader_kwargs = kwargs.get("reader", dict())
        self.writer_kwargs = kwargs.get("writer", dict())

        self.start_frame = kwargs.get("start_frame", None)
        self.stop_frame = kwargs.get("stop_frame", None)

    def start(self):
        raise DeprecationWarning(
            "Don't use StandardConverter anymore for converting videos. For now, it tends to produce memory "
            "leaks.\n Prefert the syntax : writer.copy_from(reader.frames()) or "
            "writer.copy_from(reader.sequence(start_frame,stop_frame))  "
        )

        # with Pool(processes=2) as pool:
        #     read_process = pool.apply_async(
        #         self.read,
        #         (
        #             VideoReader,
        #             self.read_queue,
        #             self.message_queue,
        #             self.start_frame,
        #             self.stop_frame,
        #         ),
        #     )
        #     # transform_process = pool.apply_async(self.transform,
        #     # (self.kwargs.pop("transform_function",_lambda_transform),self.read_queue,
        #     # self.transformed_queue,self.message_queue))
        #     write_process = pool.apply_async(self.write, (VideoWriter, self.read_queue, self.message_queue))

        #     self.last_update = time.time()
        #     self.r = self.w = 0
        #     self.max_i = 1
        #     while True:
        #         msg = self.message_queue.get()
        #         self.msg_parser(msg)
        #         if "Finished" in msg:
        #             break
        # if "Failed" in msg:
        #     print("\n" + "Conversion canceled due to error")
        # else:
        #     print("\n" + "Conversion terminated succesfully")

    def msg_parser(self, message):
        if message in ("r", "w"):
            exec(f"self.{message} += 1")
            if time.time() - self.last_update > 1:
                self.last_update = time.time()
                message = (
                    rf"Reading : {(self.r / self.max_i) * 100:.2f} % - Writing : "
                    f"{(self.w / self.max_i) * 100:.2f} %"
                )
                print(message, end="\r", flush=True)
        elif len(message) >= 7 and message[:7] == "frameno":
            self.max_i = int(message[7:])
        elif len(message) >= 3 and message[:3] == "End":
            print("\n" + message, end="", flush=True)

    def read(self, reader_class, read_queue, message_queue, start=None, stop=None):
        try:
            with reader_class(self.input_path, **self.reader_kwargs) as vid_read:
                _picky = True
                if start is None and start is None:
                    _picky = False
                if start is None:
                    start = 0
                if stop is None:
                    stop = vid_read.frames_number
                message_queue.put("frameno" + str(stop - start))

                if _picky:
                    for frame_id in range(start, stop):
                        read_queue.put(vid_read.frame(frame_id))
                        message_queue.put("r")
                else:
                    for frame in vid_read.frames():
                        read_queue.put(frame)
                        message_queue.put("r")
            read_queue.put(None)
            message_queue.put("Successfull end of read process")
        except Exception:
            read_queue.put(None)
            message_queue.put("Finished - Failed read process")
        sys.stdout.flush()

    def write(self, writer_class, read_queue, message_queue):
        try:
            with writer_class(self.output_path, **self.writer_kwargs) as vid_write:
                while True:
                    frame = read_queue.get()
                    if frame is None:
                        break
                    message_queue.put("w")
                    vid_write.write(frame)
            message_queue.put("Finished - Succesfull end of write process")
        except Exception:
            message_queue.put("Finished - Failed write process")
        sys.stdout.flush()


if __name__ == "__main__":
    test = StandardConverter(
        r"F:\\Timothe\\DATA\\BehavioralVideos\\Whisker_Video\\Whisker_Topview\\Expect_3_mush\\Mouse63\\210521_VSD1\\Mouse63_2021-05-21T16.44.57.avi",
        r"F:\\rotst.avi",
        reader={"rotate": 1},
        start_frame=300,
        stop_frame=400,
    )
    test.start()
