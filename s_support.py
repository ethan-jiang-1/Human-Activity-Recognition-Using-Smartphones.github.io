import sys
from timeit import default_timer as timer
import matplotlib.pyplot as plt

start_time=timer()

last_mark_time = None
last_mark = None
def mark_time(mark):
    global last_mark, last_mark_time
    now = timer()
    if last_mark_time is None:
        console_str = "{} occured at {:.2f} sec / {:.2f} min".format(mark, now-start_time, (now-start_time)/60)
    else:
        console_str = "{} occured at {:.2f} sec / {:.2f} min -- {:.2f} sec after last mark: {}".format(mark, now-start_time, (now-start_time)/60, now-last_mark_time, last_mark)
    print("\033[0;32m" + console_str + "\033[0;0m")
    last_mark = mark
    last_mark_time = now

def mark_milestone(mark):
    print("\033[0;33m" + mark + "\033[0;0m")
    print("\n\n")


def prompt_highlight(mark, *args):
    line = ""
    for arg in args:
        line += str(arg) + " "
    print("\033[0;36m" + str(mark)  + " " + line + "\033[0;0m")


def prompt_exception(mark, ex):
    if mark is not None:
        if mark.find("visualize_triaxial_signals") != -1:
            return
        if mark.find("look_up") != -1:
            return
    print("\033[0;31m" + str(ex) + " @" + str(mark) + "\033[0;0m")


class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations, mark=None):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40

        self.mark = mark
        if self.mark is not None:
            self.prog_bar = self.mark + ' []'
        self.cnt = 0
        self.done = False
        self.begin = timer()
        
        self.__update_amount(0)


    def inc(self, amt=1):
        if self.done:
            return        
        self.cnt += amt       
        sys.stdout.write(str(self.prog_bar) + "\r")
        sys.stdout.flush()
        self.update_iteration(self.cnt)

    def animate(self, iter):
        if self.done:
            return        
        sys.stdout.write(str(self.prog_bar) + "\r")
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        if self.done:
            return

        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        if self.mark is not None:
            self.prog_bar = self.mark + " " + self.prog_bar
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])
        if percent_done >= 100:
            print("\n")
            now = timer()
            if self.mark is not None:
                print("{} took {:.2f} sec / {:.2f} min".format(self.mark, now - self.begin, (now - self.begin)/60))
            else:
                print("took {:.2f} sec / {:.2f} min".format(now - self.begin,(now - self.begin)/60))
            self.done = True

    def __str__(self):
        return str(self.prog_bar)


def _show_nothing(*args, **kwargs):
    plt.close()

def turn_off_plt():
    print("plt is off now")
    plt.show = _show_nothing
      

flags = []
flags.append("FlSkipRaw")
flags.append("FlSkWinII")
flags.append("FlFakeAR")
def has_flag(flag):
    if flag in flags:
        return True
    return False
