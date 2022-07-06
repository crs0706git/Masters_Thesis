import time
from datetime import datetime

def record_start(mode):
    now = datetime.now()
    if mode == 'time_now':
        give_this = now.strftime("%m/%d, %H:%M:%S")
    elif mode == 'time_record':
        give_this = now.strftime("%m_%d_%H%M")
    elif mode == 'time_record_s':
        give_this = now.strftime("Month-%m_Day-%d_%H_%M_%S")
    return give_this

def start():
    time_start = time.time()
    print("Timer started")
    return time_start

def end(time_start):
    time_end = time.time()
    time_measure = time_end - time_start
    day = int(time_measure / (24 * 3600))
    hr = int((time_measure - day * 24 * 3600) / 3600)
    minute = int((time_measure - day * 24 * 3600 - hr * 3600) / 60)
    sec = int(time_measure % 60)
    return "%dday %dhr %dmin %dsec" % (day, hr, minute, sec)

def end_print(time_start):
    time_end = time.time()
    time_measure = time_end - time_start
    day = int(time_measure / (24 * 3600))
    hr = int((time_measure - day * 24 * 3600) / 3600)
    minute = int((time_measure - day * 24 * 3600 - hr * 3600) / 60)
    sec = int(time_measure % 60)

    print("%dday %dhr %dmin %dsec" % (day, hr, minute, sec))

    return day, hr, minute, sec

def pure_time(time_start):
    return time.time() - time_start

def time_is(the_time):
    day = int(the_time / (24 * 3600))
    hr = int((the_time - day * 24 * 3600) / 3600)
    minute = int((the_time - day * 24 * 3600 - hr * 3600) / 60)
    sec = int(the_time % 60)
    return day, hr, minute, sec
