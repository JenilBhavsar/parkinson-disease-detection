import time
from pynput.keyboard import Key, Listener
import csv
import argparse


class KeyLogger:
    def __init__(self , user):
        self.u_name=user
        self.press_key = 0
        self.release_key = 0
        self.hand = ''

        self.press_key_prev = 0
        self.release_key_prev = 0
        self.hand_prev = 'L'

        self.direction = ''
        self.l_hand_list = ['Q', 'W', 'E', 'R', 'T',
                            'A', 'S', 'D', 'F', 'G',
                            'Z', 'X', 'C', 'V', 'B']
        self.r_hand_list = ['Y', 'U', 'I', 'O', 'P',
                            'H', 'J', 'K', 'L', ':', ';',
                            'N', 'M', '<', '>', '.', ',', '?', '/']

        self.pass_key = 0
        self.first_iter = 1
        with Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

    def on_press(self, key):
        self.pass_key = 0
        if self.first_iter == 1:
            self.pass_key = 1
            self.first_iter = 0
        print('{0} pressed'.format(
            key))
        self.press_key = time.time()
        try:
            if key == Key.space:
                self.hand = 'S'
            elif key.char.upper() in self.l_hand_list:
                self.hand = 'L'
            elif key.char.upper() in self.r_hand_list:
                self.hand = 'R'

        except AttributeError:
            self.pass_key = 1

    def on_release(self, key):
        if key == Key.esc:
            return False
        if self.pass_key == 1:
            return
        print('{0} released'.format(
            key))
        self.release_key = time.time()

        # Logic

        ht = self.hold_time()
        lt = self.latency_time()
        ft = self.flight_time()
        gd = self.get_direction()

        with open('./user_csv/' + self.u_name + '.csv', 'a') as f:
            writer = csv.writer(f)
            row = [self.hand, ht, gd, lt, ft]
            writer.writerow(row)

        ####

        self.release_key_prev = self.release_key
        self.press_key_prev = self.press_key
        self.hand_prev = self.hand



    def hold_time(self):
        """
        Time between press and release for current key
ldfl        :return: hold time
        """
        hold = self.release_key - self.press_key
        return hold

    def latency_time(self):
        """
        Time between pressing the previous key and pressing current key.
        :return: latency time
        """
        latency = self.press_key - self.press_key_prev
        return latency

    def flight_time(self):
        """
        Time between release of previous key and press of current key.
        :return: flight time
        """
        flight = self.press_key - self.release_key_prev
        return flight

    def get_direction(self):
        """
        Direction of keys pressed
        :return: direction
        """
        direction = str(self.hand_prev) + str(self.hand)
        return direction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gets username as argument')
    parser.add_argument('username', nargs='?')
    args = parser.parse_args()
    # print(type(args))
    # print(args.username)
    KeyLogger(args.username)
    # KeyLogger('tirth')
