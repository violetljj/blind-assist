"""Lossless RGBA PNG encoding of native CARLA BGRA; no image resampling."""
import struct
import zlib
import numpy as np


def chunk(kind, data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)


def encode_bgra(data, width, height):
    rgba=np.frombuffer(data,dtype=np.uint8).reshape(height,width,4)[:,:,[2,1,0,3]]
    rows=np.zeros((height,1+width*4),dtype=np.uint8)
    rows[:,1:]=rgba.reshape(height,width*4)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,6,0,0,0))+
            chunk(b'IDAT',zlib.compress(rows.tobytes(),level=1))+chunk(b'IEND',b''))
