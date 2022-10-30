import threading
import socket
import time
from enum import Enum


class MessageType(Enum):
    Wait = 0
    Receivede = 1
    Tello_Up = 2
    Sequence = 3
    DoneMapping = 4
    DoneSequence = 5
    RedScreen = 6


class Color(Enum):
    Green = 1
    Orange = 2
    Pink = 3
    Red = 4
    Purple = 5
    Yellow = 6
    Brown = 7
    Blue = 8
    Unknown = 9


class Shape(Enum):
    Triangle = 1
    Rectangle = 2
    Square = 3
    Rhombus = 4
    Pentagon = 5
    Octagon = 6
    Star = 7
    Circle = 8
    Unknown = 9


def Message2str(MessageArry):
    return "".join(str(x) for x in MessageArry)


MessageLen = 10


class CServer:
    def __init__(self, ip, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.HOST = ip  # socket.gethostname()
        self.PORT = port  # 35004
        self.connected = False
        self.Message = None
        self.Message2Send = None

    def EstablishConnection(self):
        while not self.connected:
            try:
                self.s.bind((self.HOST, self.PORT))
                self.s.listen(5)
                self.conn, address = self.s.accept()  # Establish connection with client.
                print("Connection Established")
                self.connected = True
            except Exception as e:
                print(e)
                pass  # Do nothing, just try again

    def SendReciveThread(self):
        self.EstablishConnection()
        while self.connected:
            # Send message(if one exist) and wait for it to be accepted
            if self.Message2Send is not None:
                data = self.Message2Send.encode()
                accepted = False
                self.conn.send(data)
                while not accepted:
                    #self.conn.send(data)
                    try:
                        tempdata = self.conn.recv(MessageLen)
                        if tempdata.decode()[0] == str(MessageType.Receivede.value):
                            accepted = True
                            print("Message that was sent was Received")
                            self.Message2Send = None
                            self.Message = None
                    except Exception as e:
                        print("did not Get Received message")
                        pass

            # Try to get message, and give appropriate response
            try:
                Message = self.conn.recv(MessageLen)
                Message = Message.decode()
                if Message[0] != str(MessageType.Wait.value) and Message[0] != str(MessageType.Receivede.value):
                    print("Got : ", Message)
                    self.Message = Message
                    mess = Message2str([MessageType.Receivede.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    self.conn.send(mess.encode())
                # if Message[0] == str(MessageType.Wait.value):
                #     print("Waiting")
                if Message[0] == str(MessageType.Receivede.value):
                    print("Recieved")
            except Exception as e:
                print(e)
                pass
            mess = Message2str([MessageType.Wait.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.conn.send(mess.encode())


class Client:
    def __init__(self, ip, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # socket()
        self.HOST = ip  # socket.gethostname()
        self.PORT = port  # 35004
        self.connected = False

        self.WaitResponce = False
        self.SendResponce = True
        self.StartMapping = False
        self.StartSequence = False
        self.Message = None
        self.Message2Send = None

    def EstablishConnection(self):
        while not self.connected:
            try:
                self.s.connect((self.HOST, self.PORT))
                self.connected = True
                print("Connection Established")
            except Exception as e:
                print(e)
                pass  # Do nothing, just try again

    def SendReciveThread(self):
        self.EstablishConnection()
        mess = Message2str([MessageType.Wait.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.s.send(mess.encode())
        while self.connected:
            # Send message(if one exist) and wait for it to be accepted
            if self.Message2Send is not None:
                data = self.Message2Send.encode()
                accepted = False
                while not accepted:
                    self.s.send(data)
                    try:
                        tempdata = self.s.recv(MessageLen)
                        if tempdata.decode()[0] == str(MessageType.Receivede.value):
                            accepted = True
                            self.Message2Send = None
                    except Exception as e:
                        print(e)
                        pass
            # Try to get message, and give appropriate response
            try:
                Message = self.s.recv(MessageLen)
                Message = Message.decode()
                if Message[0] != str(MessageType.Wait.value) and Message[0] != str(MessageType.Receivede.value):
                    print("Got : ", Message)
                    self.Message = Message
                    mess = Message2str([MessageType.Receivede.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    self.s.send(mess.encode())
            except Exception as e:
                print(e)
                pass
            mess = Message2str([MessageType.Wait.value, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.s.send(mess.encode())


def main():

    ip = "192.168.0.1"
    port = 8080
    Server = CServer(ip, port)
    threadbox = threading.Thread(target=Server.SendReciveThread, args=())
    threadbox.setDaemon(True)
    threadbox.start()
    # Working on connection

    time.sleep(10)
    Server.Message2Send = Message2str([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Tello Up
    print(f'message to send {Server.Message2Send}')
    i = 0
    while True:
        if Server.Message:
            if Server.Message[0] == str(MessageType.DoneMapping.value):
                time.sleep(10)
                Server.Message2Send = Message2str([3, 1, 4, 2, 5, 7, 3, 0, 0, 0])
                print(f'message to send {Server.Message2Send}')
                Server.Message = None
            elif Server.Message[0] == str(MessageType.DoneSequence.value):
                time.sleep(10)
                i += 1
                Server.Message2Send = Message2str([3, 1, i, 0, 0, 0, 0, 0, 0, 0])
                print(f'message to send {Server.Message2Send}')
                Server.Message = None


if __name__ == '__main__':
    main()