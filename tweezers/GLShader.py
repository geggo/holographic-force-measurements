#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
#
# Copyright Tristam Macdonald 2008.
#
# Distributed under the Boost Software License, Version 1.0
# (see http://www.boost.org/LICENSE_1_0.txt)
#
''' Base shader class.

    Example:
    --------
    shader = Shader()

    shader.bind()
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(lut.target, lut.id)
    shader.uniformi('lut', 1)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(texture.target, texture.id)
    shader.uniformi('texture', 0)
    shader.uniformf('pixel', 1.0/texture.width, 1.0/texture.height)

    texture.blit(x,y,w,h)
    shader.unbind()
'''
from __future__ import print_function 
import os
import OpenGL.GL as gl
import ctypes

class Shader:
    ''' Base shader class. '''

    def __init__(self, vert = '', frag = '', geom = '', name=''):
        ''' vert, frag and geom take arrays of source strings
            the arrays will be concatenated into one string by OpenGL.'''

        self.uniforms = {}
        self.name = name
        # create the program handle
        self.handle = gl.glCreateProgram()
        # we are not linked yet
        self.linked = False
        # create the vertex shader
        self._build_shader(vert, gl.GL_VERTEX_SHADER)
        # create the fragment shader
        self._build_shader(frag, gl.GL_FRAGMENT_SHADER)
        # geometry shader 
        self._build_shader(geom, gl.GL_GEOMETRY_SHADER)
        # link the program
        self._link()

    def _build_shader(self, strings, stype):
        ''' Actual building of the shader '''

        count = len(strings)
        # if we have no source code, ignore this shader
        if count < 1:
            return

        # create the shader handle
        shader = gl.glCreateShader(stype)

        # Upload shader code
        gl.glShaderSource(shader, strings)

        # compile the shader
        gl.glCompileShader(shader)

        # retrieve the compile status
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

        # if compilation failed, print the log
        if not status:
            # display the log
            print(gl.glGetShaderInfoLog(shader))
        else:
            # all is well, so attach the shader to the program
            gl.glAttachShader(self.handle, shader)

    def _link(self):
        ''' Link the program '''

        gl.glLinkProgram(self.handle)

        # retrieve the link status
        #temp = ctypes.c_int(0)
        #gl.glGetProgramiv(self.handle, gl.GL_LINK_STATUS, ctypes.byref(temp))
        link_status = gl.glGetProgramiv(self.handle, gl.GL_LINK_STATUS)

        # if linking failed, print the log
        if link_status == gl.GL_FALSE: #not temp:
            print("Error linking shader")
            log = gl.glGetProgramInfoLog(self.handle)
            print(log)
        else:
            # all is well, so we are linked
            self.linked = True

    def bind(self):
        ''' Bind the program, i.e. use it. '''
        gl.glUseProgram(self.handle)

    def unbind(self):
        ''' Unbind whatever program is currently bound - not necessarily this
            program, so this should probably be a class method instead. '''
        gl.glUseProgram(0)

    def attrib_loc(self, name):
        return gl.glGetAttribLocation(self.handle, name)

    def uniform_loc(self, name):
        loc = self.uniforms.get(name,gl.glGetUniformLocation(self.handle,name))
        self.uniforms[name] = loc
        return loc

    def uniformf(self, name, *vals):
        ''' Uploads float uniform(s), program must be currently bound. '''

        loc = self.uniform_loc(name)

        # Check there are 1-4 values
        if len(vals) in range(1, 5):
            # Select the correct function
            { 1 : gl.glUniform1f,
              2 : gl.glUniform2f,
              3 : gl.glUniform3f,
              4 : gl.glUniform4f
              # Retrieve uniform location, and set it
            }[len(vals)](loc, *vals)

    def uniformi(self, name, *vals):
        ''' Upload integer uniform(s), program must be currently bound. '''

        loc = self.uniform_loc(name)
        # Checks there are 1-4 values
        if len(vals) in range(1, 5):
            # Selects the correct function
            { 1 : gl.glUniform1i,
              2 : gl.glUniform2i,
              3 : gl.glUniform3i,
              4 : gl.glUniform4i
              # Retrieves uniform location, and set it
            }[len(vals)](loc, *vals)

    def uniform_matrixf(self, name, mat):
        ''' Upload uniform matrix, program must be currently bound. '''

        loc = self.uniform_loc(name)
        # Upload the 4x4 floating point matrix
        gl.glUniformMatrix4fv(loc, 1, False, (ctypes.c_float * 16)(*mat))

    def uniform4fv(self, name, array):
        ''' Upload array of float4 '''
        loc = self.uniform_loc(name)
        gl.glUniform4fv(loc, array.shape[0], array)

    def uniform1fv(self, name, array):
        ''' Upload array of float '''
        loc = self.uniform_loc(name)
        gl.glUniform1fv(loc, array.shape[0], array)

def read_shader(filename):
    ''' Read a file from within the shader directory and return content '''
    
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, filename)
    fid = open(path)
    buf = fid.read()
    fid.close()
    return buf
