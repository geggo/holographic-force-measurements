#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
import traceback
import pyopencl as cl
import ctypes

import OpenGL
OpenGL.FORWARD_COMPATIBLE_ONLY = True
#OpenGL.FULL_LOGGING=True
#OpenGL.ERROR_LOGGING=True
import OpenGL.GL as gl
import logging
logging.basicConfig(level = logging.INFO)

import numpy as np

from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'
#export QT_API=pyqt

from pyface.qt import QtGui, QtCore
from pyface.qt.QtOpenGL import QGLWidget, QGLFormat

from traits.api import (HasTraits, Int, Float, Bool, Range, Str, Tuple, List, Array, Property, Instance, Python)
from traitsui.api import View, Item, TupleEditor, RangeEditor, ListEditor
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

from GLShader import Shader
from camimage import CamImage

is_qt5 = QtCore.__version_info__ >= (5,)

class GLImgWidget(QGLWidget):
    """
    with feature: fill texture directly from cl kernel, using CL/GL
    interoperability (shared texture/image)
    """

    vertex_shader_src = """
    #version 330
    uniform float zoom;
    uniform vec2 img_center;
    in vec3 vert;
    in vec2 texcoord;
    out vec2 coord;
    void main() {
        coord = texcoord;
        gl_Position = vec4((vert.x+img_center.x)*zoom, (vert.y+img_center.y)*zoom, 0.5f, 1.0f);
    }
    """
    pixel_shader_color_src = """
    #version 330
    uniform sampler2D tex;
    in vec2 coord;
    out vec4 color;
    void main() {
        vec4 t = texture(tex, coord);
        //color = texture(tex, coord);
        color = vec4(t.r, t.g, t.b, 1.0f);
    }
    """

    pixel_shader_gray_src = """
    #version 330
    uniform sampler2D tex;
    in vec2 coord;
    out vec4 color;
    void main() {
        vec4 t = texture(tex, coord);
        color = vec4(t.r, t.r, t.r, 1.0f);
    }
    """
    #texture = None

    def __init__(self, parent=None, size = (640,480), interpolation=False, bits=8): 
        #turned of interpolation for phase stepping spots (Franziska Juni 2019)
        #require OpenGL 3.2 Core
        format = QGLFormat()
        format.setSampleBuffers(True)

        if sys.platform == 'darwin':
            format.setVersion(3,2)
            format.setProfile(QGLFormat.CoreProfile)

        QGLWidget.__init__(self, format, parent)
        #print 'GL version: %d.%d'%(self.format().majorVersion(), self.format().minorVersion())

        try:
            self.pixel_ratio = self.devicePixelRatio()
        except AttributeError:
            self.pixel_ratio = 1
        
        #self.setFocusPolicy(StrongFocus)

        self.img_width, self.img_height = size
        self.interpolation = interpolation
        self.components = 1
        self.bits = bits
        self._roi = (0,0,0,0) #ROI w h x y
        self.texture = None
        self.camimage = None
        self.camimage_changed = False
        self.viewport = 0,0,0,0

        self.zoom = 1.0
        self.img_center = (0.0, 0.0) #image center: normalized image coordinates that are drawn in center of window

        self._mm_origin = None
        
        qsp = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred,
                                QtGui.QSizePolicy.Preferred
            #QtGui.QSizePolicy.MinimumExpanding,
            #QtGui.QSizePolicy.MinimumExpanding,
            #QtGui.QSizePolicy.Ignored,
            #QtGui.QSizePolicy.Ignored,
            
            )
        qsp.setHeightForWidth(True)
        self.setSizePolicy(qsp)

    def sizeHint(self):
        #recommended size, note: not strictly obeyed by qt
        return QtCore.QSize(self.img_width, self.img_height)

    def heightForWidth(self, w):
        h = int(round(w*self.img_height/self.img_width))
        #print 'height', h
        return h

    def resizeGL(self, w, h):
        #self.makeCurrent()
        aspect = self.img_height/self.img_width
        hi = int(round(w*aspect)) #ideal sizes
        wi = int(round(h/aspect))
        if hi > h:
            wt, ht = wi, h
        else:
            wt, ht = w, hi

        bx = int(0.5*(w-wt))
        by = int(0.5*(h-ht))
        gl.glViewport(bx, by, wt, ht)
        #gl.glFlush()
        self.viewport = bx, by, wt, ht #????
        
    def initializeGL(self):
        """overrides virtual function of QGLWidget, called when painting widget content is requested"""

        self.init_cl()
        self.init_gl()
        #data = self.init_data_rgb()
        #data = self.init_data_gray()
        data = self.init_data_gray_16()
        camimage = CamImage()
        camimage.data_roi = data
        self.update_texture_from_camimage(camimage)
        self.create_shared_image() #TODO: not used
        print("OpenGL initialized")

    def create_cl_context(self, gl_sharing = True):
        #global cl_context
        #if cl_context is not None:
        #    return cl_context
        #TODO: remove?????

        from pyopencl.tools import get_gl_sharing_context_properties

        #create context with GL sharing enabled
        platform = cl.get_platforms()[0]
        self.device = platform.get_devices(cl.device_type.GPU)[1]  ##### Mac fails with 0 (Iris)
        # TODO: need to find a way to get GPU device used for current OpenGL context

        if sys.platform == "darwin":
            cl_ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                                devices=[self.device,])
        else:
            try:
                cl_ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)]
                                      + get_gl_sharing_context_properties())
            except:
                print("oops")
                cl_ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)]
                                      + get_gl_sharing_context_properties(),
                                      devices = [platform.get_devices()[0]])
        return cl_ctx

    def init_cl(self):
        self.cl_ctx = self.create_cl_context()
        self.cl_queue = cl.CommandQueue(self.cl_ctx)
        
        if False:
            #build CL kernels
            with open('imgkernel.cl') as f:
                source = f.read()
                self.cl_prog = cl.Program(self.cl_ctx, source).build()
                build_info = self.cl_prog.get_build_info(self.device, cl.program_build_info.LOG)
            if build_info:
                print('build info:')
                print(build_info)

    def create_shared_image(self):
        #create shared cl_image from texture
        self.cl_image_shared = cl.GLTexture(context = self.cl_ctx, 
                                            flags = cl.mem_flags.WRITE_ONLY, 
                                            texture_target = gl.GL_TEXTURE_2D,
                                            miplevel = 0,
                                            texture = self.texture,
                                            dims = 2)
        
    def init_texture(self, w, h, c, bits):
        """(re)init texture, no upload"""
        self.img_width, self.img_height = w, h
        self.components = c
        self.bits = bits
        self.format = {1: gl.GL_RED,
                       3: gl.GL_BGR,
                       4: gl.GL_BGRA}[self.components]
        
        if self.texture is not None:
            gl.glDeleteTextures([self.texture])
        
        ## texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        #create empty GL texture: RGBA 8bit
        #TODO: float????
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, 
                        gl.GL_RGBA, self.img_width, self.img_height, 0, 
                        gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE if bits <=8 else gl.GL_UNSIGNED_SHORT,
                        #gl.GL_UNSIGNED_BYTE,
                        None)
        #set texture parameters
        gl.glTexParameter(gl.GL_TEXTURE_2D, #texture filter for minification
                          gl.GL_TEXTURE_MIN_FILTER, 
                          gl.GL_LINEAR_MIPMAP_LINEAR,
                          )
        
        gl.glTexParameter(gl.GL_TEXTURE_2D, #texture filter for magnification
                          gl.GL_TEXTURE_MAG_FILTER,
                          gl.GL_LINEAR if self.interpolation else gl.GL_NEAREST
                          ) 

        gl.glGenerateMipmap(gl.GL_TEXTURE_2D) #generate mipmaps

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


    def init_gl(self):
        #enable multisampling
        gl.glEnable(gl.GL_MULTISAMPLE)

        #gl.glDisable(gl.GL_DEPTH_TEST)

        #print "samples", gl.glGetInteger(gl.GL_SAMPLES)   

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        #cl writes into image (shared with texture), used in shader

        #set mipmap parameters
        #gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        #gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
        
        ##sampler
        #self.sampler = gl.glGenSamplers(1)
        #gl.glSamplerParameter(self.sampler, gl.GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        #gl.glSamplerParameter(self.sampler, gl.GL_TEXTURE_MAG_FILTER, GL_NEAREST)


        #shader
        self.shader_color = Shader(self.vertex_shader_src,
                                   self.pixel_shader_color_src)
        self.shader_gray = Shader(self.vertex_shader_src,
                                  self.pixel_shader_gray_src)
        for s in (self.shader_color, self.shader_gray):
            s.bind()
            s.uniformi('tex', 0)
            s.unbind()
        
        #vertex and texture coordinates buffers
        self.vertices = np.array([-1.0,  1.0, 0.5,
                                  -1.0, -1.0, 0.5, 
                                   1.0, -1.0, 0.5,
                                   1.0,  1.0, 0.5,
                                   ],
                                 dtype = np.float32)

        self.texcoords = np.array([0.0, 0.0,
                                   0.0, 1.0,
                                   1.0, 1.0,
                                   1.0, 0.0,
                                   ],
                                  dtype = np.float32)
        
        self.vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vertex_array)

        self.vertex_buffer = gl.glGenBuffers(1)
        self.texcoords_buffer = gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                        size = self.vertices.nbytes, 
                        data = self.vertices.ctypes.data_as(ctypes.c_void_p),
                        usage = gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.shader_color.attrib_loc('vert'), #TODO: gray too
                                 3, #vec2
                                 gl.GL_FLOAT, #type
                                 gl.GL_FALSE, #normalize
                                 4*3, #stride
                                 ctypes.c_void_p(0)
                                 ) #offset
        gl.glEnableVertexAttribArray(self.shader_color.attrib_loc('vert'))
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texcoords_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        size = self.texcoords.nbytes,
                        data = self.texcoords.ctypes.data_as(ctypes.c_void_p),
                        usage = gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.shader_color.attrib_loc('texcoord'),
                                 2,
                                 gl.GL_FLOAT,
                                 gl.GL_FALSE,
                                 4*2,
                                 ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.shader_color.attrib_loc('texcoord'))
        gl.glBindVertexArray(0)
        

    def set_camimage(self, camimage):
        self.camimage = camimage
        self.camimage_changed = True

    #def set_imgdata(self, img):
    #    self.set_camimage(CamImage(img))

    def update_texture_from_camimage(self, camimg):
        #self.makeCurrent() #not necessary if called within paintGL
        camimg_shape = (camimg.w, camimg.h, camimg.c, camimg.bits)
        texture_shape = (self.img_width, self.img_height, self.components, self.bits)
        if self.texture is None  or  texture_shape != camimg_shape:
            #need (re)init
            self.init_texture(camimg.w, camimg.h, camimg.c, camimg.bits)
            self.update_texture_from_camimage(camimg)  # TODO: dangerous recursive call

            #resize viewport
            s = self.size()
            self.resizeGL(s.width(), s.height())
            #self.updateGL() #TODO????
        else: #update existing texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            camimg_roi = (camimg.roi_w, camimg.roi_h)+camimg.pos
            if camimg_roi == self._roi: 
                #ROI unchanged, only update ROI data
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D,
                                   0, #level
                                   camimg.pos[0], camimg.pos[1], camimg.roi_w, camimg.roi_h,
                                   self.format,
                                   gl.GL_UNSIGNED_BYTE if self.bits <= 8 else gl.GL_UNSIGNED_SHORT,
                                   camimg.data_roi)
            else: #update full texture and _roi cache
                self._roi = camimg_roi
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D,
                                   0, #level
                                   0,0,camimg.w,camimg.h,
                                   self.format,
                                   gl.GL_UNSIGNED_BYTE if self.bits <= 8 else gl.GL_UNSIGNED_SHORT,
                                   camimg.data_fullsize)
        
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def calculate_texture_cl(self):
        # update texture from cl
        cl.enqueue_acquire_gl_objects(self.cl_queue, [self.cl_image_shared])# .wait()
        self.cl_prog.fill_texture(self.cl_queue,
                                  (self.img_width, self.img_height),
                                  #(64, 1),
                                  (8,8),
                                  self.cl_image_shared) # .wait()
        cl.enqueue_release_gl_objects(self.cl_queue, [self.cl_image_shared]).wait()

    def update_texture_cl(self):
        self.calculate_texture_cl()
        # gl.glFlush()  # TODO: needed? (before?)

        # regenerate mipmap
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.update()  # request redraw

    def paintGL(self):
        """overrides virtual function of QGLWidget, called when painting widget content is requested"""
        # print('entered paintGL')

        if self.camimage_changed:
            try:
                self.update_texture_from_camimage(self.camimage)
            except Exception as e:
                print("ignored exception in paintGL")
                exc_type, exc_value, exc_tb = sys.exc_info()
                info = str(e)
                print('\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)[2:]))

            self.camimage_changed = False

        try:
            gl.glClearColor(0.25, 0.25, 0.25, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # necessary for lab computers

            if self.components == 1:
                shader = self.shader_gray
            else:
                shader = self.shader_color

            shader.bind()

            #setting shader vars
            shader.uniformf('zoom', self.zoom)
            shader.uniformf('img_center', *self.img_center)

            #gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE);
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            #gl.glBindSampler( 0, self.sampler)
            gl.glBindVertexArray(self.vertex_array)
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
            gl.glBindVertexArray(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            shader.unbind()
        except gl.GLError as e:
            print('caught GLError in paintGL, ignoring', e)

        # print('leaving paintGL')

    def init_data_rgb(self):
        data = np.zeros((self.img_height, self.img_width, 4), np.uint8) #rows, columns
        data[256:, 256:, 0] = 255
        data[:256, 256:, 1] = 255
        data[256] = 255
        data[:,256] = 255
        data[:100, :50:2, :3] = 255
        data[:,:,3] = 128 #alpha ?
        return data

        
    def init_data_gray(self):
        data = np.zeros((480, 640), np.uint8) #rows, columns
        data[:32:2, :32:2] = 255
        data[:32:2, -32:-1:2] = 128
        data[200] = 255
        data[300] = 200
        data[:,256] = 255
        return data

    def init_data_gray_16(self):
        data = np.zeros((480, 640), np.uint16) #rows, columns
        data[:32:2, :32:2] = 40000
        data[:32:2, -32:-1:2] = 40000
        data[200] = 30000
        data[300] = 30000
        data[:,256] = 30000
        return data


    def wheelEvent(self, event):
        if is_qt5:
            if event.pixelDelta().y() > 0 or event.angleDelta().y() > 0:
                self.zoom *= 1.2
            elif event.pixelDelta().y() < 0 or event.angleDelta().y() < 0:
                self.zoom /= 1.2
        else:
            if event.delta() > 0:
                self.zoom *= 1.2
            elif event.delta() < 0:
                self.zoom /= 1.2
        self.zoom = np.clip(self.zoom, (1./1.2)**10, 1.2**20)
        if abs(self.zoom - 1.) < 0.05:
            self.zoom = 1.

        self.do_update('wheel Event')
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == 1:
            self._mm_origin = event.x(), event.y()
            self._img_center = (self.img_center[0], self.img_center[1])
            event.ignore()
        else:
            print('screen pos', (event.x(), event.y()), 'img pos', self.screen_to_img(event.x(), event.y()))
            #pass

    def mouseMoveEvent(self, event):
        if self._mm_origin is not None:
            x, y = event.x(), event.y()
            viewport_width, viewport_height = self.viewport[2], self.viewport[3]
            self.img_center =  (self._img_center[0] + 2*self.pixel_ratio*(x-self._mm_origin[0])/(viewport_width*self.zoom),
                                self._img_center[1] - 2*self.pixel_ratio*(y-self._mm_origin[1])/(viewport_height*self.zoom))
            #TODO: temporarily removed
            self.do_update('mouseMoveEvent')
            event.ignore()

    def mouseReleaseEvent(self, event):
        self._mm_origin = None

    def screen_to_img(self, xs, ys):
        #convert screen coordinates to image coordinates
        #note: screen relative to widget top left
        #normalized pos. rel. viewport. top left (-1, -1), bottom right (1, 1)
        
        xv = (2*(xs-self.viewport[0])*self.pixel_ratio/self.viewport[2] - 1) #normalized pos rel. viewport
        yv = (2*(ys-self.viewport[1])*self.pixel_ratio/self.viewport[3] - 1) #normalized pos rel. viewport
        xi = xv/self.zoom - self.img_center[0] #pos rel image
        yi = yv/self.zoom + self.img_center[1] #pos rel image
        return xi, yi

    def do_update(self, context = 'no context given'):
        #threadname = threading.current_thread().name
        #if threadname is 'MainThread':
        #    self.update()
        #else:
        #    print 'do update not in MainThread', context
        self.update()
        

class SLMWidgetGL(GLImgWidget):

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, self.img_width, self.img_height)
        self.viewport = (0, 0, self.img_width, self.img_height)

    def wheelEvent(self, event):
        pass

        
class SLMWindow(QtGui.QMainWindow):
    def __init__(self, holosize=(1920, 1080 ), secondary_screen = False):
        QtGui.QMainWindow.__init__(self)
        #window = QtGui.QWidget()
        self.slmwidget = SLMWidgetGL(None, size=holosize, interpolation=False)
        #layout = QtGui.QHBoxLayout()
        #layout.addWidget(self.imgwidget, 0, QtCore.Qt.AlignCenter)
        #window.setLayout(layout)
        #self.setCentralWidget(window)
        self.setCentralWidget(self.slmwidget)
        self.setWindowTitle('OpenGL image widget')

        if secondary_screen:
            #TODO: automatically detect SLM screen (from size?)
            self.move(2560, 0)
            #self.showMaximized()
            self.setWindowState(QtCore.Qt.WindowFullScreen)

        data = np.zeros( (holosize[1], holosize[0]), dtype=np.uint8)
        data[:500:2,:500:2] = 255
        self.slmwidget.set_camimage(CamImage(data))
        self.show()
        

        
class _GLImageEditor(Editor):
    """Editor for displaying image, uses GLImgWidget from glimgwidget"""
    scrollable = True
    #inherited attributes:
    #.value: class Image
    #.control: class GLImgWidget
    #inherited methods (to be implemented): update_editor(), rebuild_editor(), update_object() ?
    
    def init(self, parent):
        #self.sync_value(self.factory.xxx_name, 'trait_name', 'from')
        self.control = GLImgWidget()
        self.set_tooltip()

    def update_editor(self):
        self.control.set_camimage(self.value)
        self.control.do_update('_GLImageEditor:updateEditor')
        
class GLImageEditor(BasicEditorFactory):
    klass = _GLImageEditor

    
class GLOverlay(object):
    visible = True
    
    def __init__(self):
        self.initialized_GL = False
    
    def initGL(self):
        pass

    def paintGL(self, zoom, origin):
        raise Exception('Not implemented')
    
    def paint(self, zoom, origin):
        if not self.visible:
            return
        if not self.initialized_GL:
            #self.initGL()
            print('gloverlay not yet initialized!')

        self.paintGL(zoom, origin)

    def update_overlay(self, overlay_trait, img_size):
        pass

class GLOverlayPolygon(GLOverlay):
    scale = [400/640., 400/480.]
    center = [0.1, 0.]
    angle = 0.
    edge_color = [1.0, 1.0, 0.0, 1.0]
    
    GL_LINE = gl.GL_LINE_STRIP 

    vertex_shader_src = """
    #version 150
    uniform float zoom;
    uniform vec2 origin;
    uniform vec2 center;
    uniform vec2 scale;
    uniform float angle;
    in vec3 vert;
    out vec2 coord;
    void main() {
    vec2 p = vec2( vert.x * cos(angle) - vert.y * sin(angle), vert.x * sin(angle) + vert.y*cos(angle));
    gl_Position = vec4( zoom*(scale.x*p.x - center.x + origin.x), zoom*(scale.y*p.y - center.y + origin.y), 0.5f, 1.0f);
    }
    """

    pixel_shader_src = """
    #version 150
    uniform vec4 edge_color;
    out vec4 color;
    void main() {
        //color = vec4(1.0f, 0.0f, 0.0f, 0.5f);
        color = edge_color;
    }
    """

    def initGL(self):

        self.init_vertices()
        
        self.shader = Shader(self.vertex_shader_src, self.pixel_shader_src)
        self.vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vertex_array)
        self.vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                        size = self.vertices.nbytes,
                        data = self.vertices.ctypes.data_as(ctypes.c_void_p),
                        usage = gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.shader.attrib_loc('vert'),
                                 3, #vec3
                                 gl.GL_FLOAT, #type
                                 gl.GL_FALSE, #normalize
                                 4*3, #stride
                                 ctypes.c_void_p(0)
                                 )
        gl.glEnableVertexAttribArray(self.shader.attrib_loc('vert'))
        gl.glBindVertexArray(0)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        #print gl.glGet(gl.GL_SMOOTH_LINE_WIDTH_RANGE)
        
        self.initialized_GL = True

    def paintGL(self, zoom, origin):
        shader = self.shader
        shader.bind()
        shader.uniformf('origin', *origin)
        shader.uniformf('zoom', zoom)
        shader.uniformf('angle', self.angle*np.pi/180)
        shader.uniformf('scale', *self.scale)
        shader.uniformf('center', *self.center)
        shader.uniformf('edge_color', *self.edge_color)

        gl.glBindVertexArray(self.vertex_array)
        lw = zoom if zoom < 1.0 else 1.
        gl.glLineWidth(lw) #TODO: ?????
        gl.glDrawArrays(self.GL_LINE, 0, self.n_vertices)
        gl.glBindVertexArray(0)
        shader.unbind()


class GLOverlayCircle(GLOverlayPolygon):
    
    def init_vertices(self):
        self.n_vertices = n = 64
        alpha = np.arange(n, dtype=np.float32)*(2*np.pi/(n-1))
        self.vertices = np.empty((n,3), dtype = np.float32)
        self.vertices[:,0] = np.cos(alpha)
        self.vertices[:,1] = np.sin(alpha)
        self.vertices[:,2] = 0.6
        self.vertices.shape = (-1,)
    
    def update_overlay(self, overlay_trait, img_size):
        nx, ny = img_size
        o = overlay_trait
        self.visible = o.visible
        self.angle = o.angle
        self.scale = [2*o.radius/nx, 2*o.radius/ny]
        self.center = [-((o.center[0] + 0.5)*2./nx - 1.),
                        ((o.center[1] + 0.5)*2./ny - 1.)]
        
class GLOverlayRectangle(GLOverlayPolygon):

    scale = [30./640, 100./480]
    center = [100/640., 
              (0.5*(200+300 + 1)/480.-0.5)*2
              ]
    edge_color = [0.0, 1.0, 0.0, 0.9]
    
    def init_vertices(self):
        z0 = 0.3
        self.vertices = np.array([-1.0,  1.0, z0,
                                  -1.0, -1.0, z0, 
                                   1.0, -1.0, z0,
                                   1.0,  1.0, z0,
                                   -1.0,  1.0, z0,
                                   ],
                                 dtype = np.float32)
        self.n_vertices = 5

    def update_overlay(self, overlay_trait, img_size):
        nx, ny = img_size
        o = overlay_trait
        self.visible = o.visible
        self.angle = o.angle

        self.scale = [o.size[0]*1./nx, o.size[1]*1./ny]
        self.center = [-((o.center[0]+0.5)*2./nx - 1.),
                        ((o.center[1]+0.5)*2./ny - 1.)]

class GLOverlayGrid(GLOverlayRectangle):
    edge_color = [0.0, 0.7, 0.7, 0.4]
    GL_LINE = gl.GL_LINES

    def init_vertices(self):
        n = 16+1
        v = np.zeros((4*n, 3), np.float32)
        p = np.linspace(-1,1,n)
        v[0:2*n:2, 0] = p
        v[1:2*n:2, 0] = p
        v[0:2*n:2, 1] = -1
        v[1:2*n:2, 1] = 1

        v[2*n::2, 0] = -1
        v[2*n+1::2, 0] = 1
        v[2*n::2, 1] = p
        v[2*n+1::2, 1] = p

        v[:,2] = 0.45

        self.vertices = v
        self.n_vertices = v.shape[0]
        self.vertices.shape = (-1,)

class GLOverlayArrow(GLOverlayPolygon):
    scale = [0, 0]
    center = [0.1, 0.2]
    edge_color = [0.0, 0.25, 0.75, 0.75]

    def init_vertices(self):
        z0 = 0.5
        self.vertices = np.array([ 0.0, 0.0, z0,
                                   0.0, 0.1, z0,
                                   0.9, 0.1, z0,
                                   1.0, 0.0, z0,
                                   0.9,-0.1, z0,
                                   0.0,-0.1, z0,
                                   0.0, 0.0, z0,
                                   ], dtype = np.float32)
        self.n_vertices = 7

    def update_overlay(self, overlay_trait, img_size):
        nx, ny = img_size
        o = overlay_trait
        self.visible = o.visible
        self.angle = np.arctan2(o.direction[0], o.direction[1])*(180./np.pi)
        length = np.hypot(o.direction[0], o.direction[1])
        self.scale = [length*1./nx, length*1./ny]
        self.center = [-((o.center[0]+0.5)*2./nx - 1.),
                        ((o.center[1]+0.5)*2./ny - 1.)]


class GLOverlaySpots(GLOverlay):
    spots = np.zeros( (10,3), np.float32)
    spots[0] = [0.5,0.,0.]
    colors = np.zeros( (10, 4), np.float32)
    edge_color = [1.0, 0.0, 0.0, 0.5]
    spots_changed = True
    scale = (0.01, 0.01)
    
    
    vertex_shader_src = """
    #version 330
    struct vData
    {
       vec4 color;
    };
    uniform float zoom;
    uniform vec2 origin;
    in vec3 vert;
    in vec4 colors;
    out vData vertices;
    void main() 
    {
       vec2 p = zoom*(vec2( vert.x, vert.y) + origin);
       gl_Position = vec4(p.x, p.y, 0.5f, 1.0f);
       vertices.color = colors; //vec4(0.f, 0.f, 1.f, .5f);
    }
    """

    geom_shader_src = """
    #version 330

    #define n 21
    #define k 2*3.141592f/(n-1)
    layout(points) in;
    layout(line_strip, max_vertices = n) out;
    
    struct vData
    {
       vec4 color;
    };

    uniform float zoom;
    uniform vec2 scale;
    in vData vertices[];
    out vData frag;

    void main()
    {
       int i;
       for(i = 0; i < n; i++)
       {
          gl_Position = gl_in[0].gl_Position + zoom*vec4(scale.x*cos(i*k), scale.y*sin(i*k), 0.f, 0.f);
          frag.color = vertices[0].color;
          EmitVertex();
       }
    EndPrimitive();
    }
    """

    pixel_shader_src = """
    #version 330
    
    struct vData
    {
       vec4 color;
    };

    in vData frag;
    uniform vec4 edge_color;
    out vec4 color;
    void main() {
        color = edge_color;
        color = frag.color;
    }
    """


    def initGL(self):

        self.init_vertices()
        self.init_colors()
        
        self.shader = Shader(self.vertex_shader_src, self.pixel_shader_src, self.geom_shader_src)
        self.vertex_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vertex_array)

        self.vertex_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                        size = self.vertices.nbytes,
                        data = self.vertices.ctypes.data_as(ctypes.c_void_p),
                        usage = gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(self.shader.attrib_loc('vert'),
                                 3, #vec3
                                 gl.GL_FLOAT, #type
                                 gl.GL_FALSE, #normalize
                                 4*3, #stride
                                 ctypes.c_void_p(0)
                                 )
        gl.glEnableVertexAttribArray(self.shader.attrib_loc('vert'))

        self.colors_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.colors_buffer)
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,
                        size = self.colors.nbytes,
                        data = self.colors.ctypes.data_as(ctypes.c_void_p),
                        usage = gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(self.shader.attrib_loc('colors'),
                                 4, gl.GL_FLOAT, gl.GL_FALSE, 4*4, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(self.shader.attrib_loc('colors'))

        gl.glBindVertexArray(0)

        #gl.glEnable(gl.GL_POINT_SMOOTH)
        self.initialized_GL = True

    def init_vertices(self):
        self.n_vertices = self.spots.shape[0]
        self.vertices = self.spots.view()
        self.vertices[:,0] = np.arange(self.n_vertices)*0.1
        self.vertices.shape = (-1,)

    def init_colors(self):
        c = self.colors
        c[0] = [1.0, 0.0, 0.0, 1.0]
        c[1] = [1.0, 1.0, 0.0, 1.0]
        c[2] = [0.0, 1.0, 0.0, 1.0]
        c[3] = [0.0, 1.0, 1.0, 1.0]
        c[4] = [0.0, 0.0, 1.0, 1.0]
        c[5:] = [1.0, 0.0, 1.0, 0.5]

        self.colors.shape = (-1,)

    def paintGL(self, zoom, origin):
        if self.spots_changed:
            #update spots
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vertex_buffer)
            gl.glBufferSubData(target=gl.GL_ARRAY_BUFFER,
                               offset = 0,
                               size = 3*4*self.n_vertices,
                               data = self.vertices.ctypes.data_as(ctypes.c_void_p),
                               )
            
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.colors_buffer)
            gl.glBufferSubData(target=gl.GL_ARRAY_BUFFER,
                               offset = 0,
                               size = 4*4*self.n_vertices,
                               data = self.colors.ctypes.data_as(ctypes.c_void_p),
                               )
        
        shader = self.shader
        shader.bind()
        shader.uniformf('origin', *origin)
        shader.uniformf('zoom', zoom)
        shader.uniformf('scale', *self.scale)
        shader.uniformf('edge_color', *self.edge_color)

        gl.glPointSize(10*zoom)
        gl.glBindVertexArray(self.vertex_array)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.n_vertices)
        gl.glBindVertexArray(0)
        shader.unbind()

    def update_overlay(self, overlay_trait, img_size):
        nx, ny = img_size
        o = overlay_trait
        self.visible = o.visible
        self.angle = o.angle

        for k, pos in enumerate(o.spots_position):
            x, y = pos
            self.spots[k, 0:2] = [((x+0.5)*2./nx - 1.), -((y+0.5)*2./ny - 1.)]
        self.n_vertices = len(o.spots_position)
        self.spots_changed = True
        self.scale = (20/nx, 20/ny)


class GLOverlayWarpArrows(GLOverlay):
    """shows arrows indicating distortions of BFP image"""
    edge_color = [0.25, 0.75, 0.0, 0.75]
    
    data_length = 17*17 #TODO: initialize in initGL from self.data
    data = np.zeros( (17,17, 4), np.float32) #data for each spot: idx_x, idx_y, dx, dy
    data_changed = True
    
    scale = (0.05, 0.05)
    grid_origin_normalized = (0., 0.) #
    grid_lattice_vector1 = (20., 0.)
    grid_lattice_vector2 = (0., 20.)
    
    #Vertex Shader: apply image zoom and origin to overlays
    vertex_shader_src = """
    #version 330
    uniform vec2 origin; //origin image area?
    uniform float zoom;
    uniform vec2 grid_origin_normalized;
    uniform vec2 grid_lattice_vector1;
    uniform vec2 grid_lattice_vector2;
    uniform vec2 scale;

    in vec2 idx;
    in vec2 displacement;

    out vec2 arrow; //output to geometry shader: transformed displacement in normalized coordinates
    
    void main() {
    
    vec2 g1 = grid_lattice_vector1;
    vec2 g2 = -grid_lattice_vector2;

    vec2 p = - zoom * vec2(g1.x*idx.x + g2.x*idx.y, g1.y*idx.x + g2.y*idx.y);
    vec2 o = zoom * (origin  - grid_origin_normalized);
    gl_Position = vec4(scale.x*p.x + o.x, scale.y*p.y + o.y, 0.6f, 1.0f);
    arrow = zoom*scale*vec2(displacement.x, displacement.y);
    }
    """
    
    geom_shader_src = """
    #version 330
    #define n 17
    #define k 2*3.141592f/(n-1)
    layout(points) in; //struct gl_in[]
    layout(line_strip, max_vertices = 19) out;
    
    in vec2 arrow[]; //input from vertex shader

    uniform vec2 scale;
    uniform float zoom;
    void main()
    {
       vec2 pos = vec2(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y);
       vec2 head = pos + vec2(arrow[0].x,  arrow[0].y);

       gl_Position = vec4(pos.x, pos.y, 0.f, 1.f);
       EmitVertex();
       gl_Position = vec4(head.x, head.y, 0.f, 1.f);
       EmitVertex();
       EndPrimitive();
    
       //circle around head (spot position)
       int i;
       for(i = 0; i < n; i++)
       {
          gl_Position = vec4(head.x + 3*zoom*scale.x * cos(i*k), 
                             head.y + 3*zoom*scale.y * sin(i*k), 
                             0.f, 1.f);
          EmitVertex();
       }
       EndPrimitive();
    }
    """

    pixel_shader_src = """
    #version 330
    uniform vec4 edge_color;
    out vec4 color;
    void main() {
       color = edge_color;
    }
    """

    
    def initGL(self):
        #init vertices
        #self.n_arrows = 17*17; #TODO: take from self.arrow_data
        #TODO: replace by meshgrid
        idx_x, idx_y = np.meshgrid(np.arange(17)-8, np.arange(17)-8)
        self.data[...,0] = idx_x
        self.data[...,1] = idx_y

        self.data_flat = self.data.view()
        self.data_flat.shape = (-1,) #nötig?

        #compile shaders
        self.shader = Shader(self.vertex_shader_src, self.pixel_shader_src, self.geom_shader_src)

        #init vertex buffer objects
        self.data_gl_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.data_gl_buffer)
        gl.glBufferData(target = gl.GL_ARRAY_BUFFER,
                        size = self.data_flat.nbytes,
                        data = self.data_flat.ctypes.data_as(ctypes.c_void_p), #TODO: also self.data possible?
                        usage = gl.GL_DYNAMIC_DRAW) #TODO: static ?

        #init vertex array and vertex attributes
        self.data_gl_array = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.data_gl_array)
        
        #data: for each point: index (idx_x, idx_y), displacement dx, dy
        gl.glEnableVertexAttribArray(self.shader.attrib_loc('idx'))
        gl.glVertexAttribPointer(self.shader.attrib_loc('idx'),
                                 2, #vec2
                                 gl.GL_FLOAT, #type
                                 gl.GL_FALSE, #normalize
                                 4*4, #stride in bytes
                                 ctypes.c_void_p(0) #offset in bytes
                                 )
        gl.glEnableVertexAttribArray(self.shader.attrib_loc('displacement'))
        gl.glVertexAttribPointer(self.shader.attrib_loc('displacement'),
                                 2, #vec
                                 gl.GL_FLOAT, #type
                                 gl.GL_FALSE, #normalize
                                 4*4, #stride
                                 ctypes.c_void_p(2*4), #offset in bytes
                                 )
                                            
        gl.glBindVertexArray(0)
        self.initialized_GL = True

    def paintGL(self, zoom, origin):
        if self.data_changed:
            #update arrow data
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.data_gl_buffer)
            gl.glBufferSubData(target=gl.GL_ARRAY_BUFFER,
                               offset=0,
                               size=self.data_flat.nbytes, #4*4*self.data_length, #number of arrows, 4 floats per arrow, TODO: init from data
                               data = self.data_flat.ctypes.data_as(ctypes.c_void_p), #TODO: self.data ?
                               )
            shader = self.shader
            shader.bind()

            #vertex shader
            shader.uniformf('origin', *origin)
            shader.uniformf('zoom', zoom)
            shader.uniformf('grid_origin_normalized', *self.grid_origin_normalized) #center arrow ensemble in normalized coordinates 
            shader.uniformf('grid_lattice_vector1', *self.grid_lattice_vector1)
            shader.uniformf('grid_lattice_vector2', *self.grid_lattice_vector2)
            shader.uniformf('scale', *self.scale) #scaling factors from pixel to normalized coordinates
            
            #geometry shader
            
            #pixel shader
            shader.uniformf('edge_color', *self.edge_color)

            gl.glBindVertexArray(self.data_gl_array)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.data_flat.shape[0])
            gl.glBindVertexArray(0)
            shader.unbind()

    def update_overlay(self, overlay_trait, img_size):
        #take over data from overlay trait model
        nx, ny = img_size
        o = overlay_trait
        
        self.visible = o.visible
        self.grid_origin_normalized = [-((o.grid_origin[0]+0.5)*2./nx - 1.), #origin grid in normalized coordinates
                                        ((o.grid_origin[1]+0.5)*2./ny - 1.)]
        self.grid_lattice_vector1 = (-o.grid_lattice_vector1[0], o.grid_lattice_vector1[1]);
        self.grid_lattice_vector2 = (-o.grid_lattice_vector2[0], o.grid_lattice_vector2[1]);

        D = o.displacements #.view()
        self.data[:,:,2] =  -D[:,:,0]
        self.data[:,:,3] =  D[:,:,1] 
        
        self.scale = (2./nx, 2./ny) #scaling pixel to normalized coordinates (-1...1)
        self.data_changed = True
                                                        
        
class GLImgWithOverlaysWidget(GLImgWidget):

    def __init__(self, *args, **kwargs):
        GLImgWidget.__init__(self, *args, **kwargs)
        self.gloverlays = dict()

    #override
    def init_gl(self):
        GLImgWidget.init_gl(self)
        for o in self.gloverlays.values():
            o.initGL()

    #override        
    def paintGL(self):
        GLImgWidget.paintGL(self)
        for o in self.gloverlays.values():
            try:
                o.paint(zoom=self.zoom, origin=self.img_center)
            except gl.GLError as e:
                print('caught GLError in paintGL of overlay, ignoring')
            
class OverlayTrait(HasTraits):
    idx = Int(0)
    angle = Float(0.) #Range(-45., 45., 0)
    visible = Bool(True)

    def handle_mouse(self, mx, my, event_type):
        #event_type: 'left_click', 'left_drag', 'drag', ...
        #mx, my: mouse coordinates, in image coordinates (image pixels relative top left)
        pass

class OverlayTraitCircle(OverlayTrait):
    center = Tuple(Range(0, 2000, 500), Range(0, 1200, 500))#Int(300), Int(400))
    radius = Range(0, 1000, 100)
    trait_view = View(Item('center'), #editor = TupleEditor(editors = [RangeEditor(), RangeEditor()])),
                      Item('radius'),
                      )

class OverlayTraitRectangle(OverlayTrait):
    def __init__(self, **kwargs):
        OverlayTrait.__init__(self, **kwargs)
        self._mouse_interactive = False, (0,0)

    size = Tuple(Range(0, 2000, 100), Range(0, 2000, 100))
    center = Tuple(Int, Int)
    trait_view = View(
        Item('visible'),
        Item('center', editor = TupleEditor(labels = ['x', 'y'])),
        Item('size', editor = TupleEditor(labels = ['w', 'h'])),
        Item('angle', editor = RangeEditor(low=-90, high=90)),
        )
    _mouse_interactive = Python

    def handle_mouse(self, mx, my, event_type):
        #called by filter function, return True if event is accepted (skip event)
        if event_type is 'left_click':
            posx, posy = self.center
            if (abs(posx-mx) + abs(posy-my)) < 20:
                #print "hit rectangle center"
                self._mouse_interactive = (True, (posx-mx, posy-my))
                return True
        elif event_type is 'drag':
            active,(dx, dy)  = self._mouse_interactive
            if active:
                self.center = (int(mx+dx), int(my+dy))
                return True
        elif event_type is 'release':
            active, (dx, dy) = self._mouse_interactive
            if active:
                self._mouse_interactive = (False, (0,0))
                return True
        return False

class OverlayTraitGrid(OverlayTrait):
    extend = Range(0,2000,500)
    size = Property(Tuple(Float, Float), depends_on = 'extend')
    center = Tuple(Float(100), Float(100))
    trait_view = View(
        Item('visible'),
        Item('angle', editor = RangeEditor(low=-45., high=45.)),
        Item('center', editor = TupleEditor(labels = ['x', 'y'])),
        #Item('size', editor = TupleEditor(labels = ['w', 'h'])),
        Item('extend'),
        )

    def _get_size(self):
        return (self.extend, self.extend)

class OverlayTraitSpots(OverlayTrait):

    def __init__(self, **kwargs):
        OverlayTrait.__init__(self, **kwargs)
        self._spot_interactive = False, 0, (0,0)

    spots_position = List(
        Tuple(
            Float, #Range(0.,1000., 300.),
            Float, #Range(0.,1000., 300.),
            ),
        value = [(100., 0),
                 (200., 10),
                 (300., 20),
                 (400., 30),
        ],
        )

    _spot_interactive = Python#( default_value = (False,0,(0,0))) 
    
    trait_view = View(
        Item('spots_position',),#editor = TupleEditor(labels = ['x', 'y'])),
        Item('visible'),
        )

    def handle_mouse(self, mx, my, event_type):
        #called by filter function, return True if event is accepted (skip event)
        if event_type is 'left_click':
            for k, (posx, posy) in enumerate(self.spots_position):
                if (abs(posx-mx) + abs(posy-my)) < 10:
                    #print "hit spot #", k
                    self._spot_interactive = (True, k, (posx-mx, posy-my))
                    return True
        elif event_type is 'drag':
            active,k,(dx, dy)  = self._spot_interactive
            if active:
                self.spots_position[k] = (mx+dx, my+dy)
                return True
        elif event_type is 'release':
            active, k, (dx, dy) = self._spot_interactive
            if active:
                self._spot_interactive = (False, 0, (0,0))
                return True
        return False

class OverlayTraitArrow(OverlayTrait):
    center = Tuple(Float(100), Float(100))
    direction = Tuple(Float(5.), Float(5.), Float(5.))
    trait_view = View(
        Item('visible'),
        Item('center',
             editor = TupleEditor(
                 labels = ['x', 'y'],
                 editors = [RangeEditor(low=0, high=2000),
                            RangeEditor(low=0, high=2000)],
                 )
                 ),
        Item('direction',
             editor = TupleEditor(editors = [RangeEditor(low=0, high=50, mode='slider'),
                                             RangeEditor(low=0, high=50, mode='slider'),
                                             RangeEditor(low=0, high=50, mode='slider'),],),),)

class OverlayTraitWarpArrows(OverlayTrait):
    grid_origin = Tuple(Float(100), Float(100))
    grid_lattice_vector1 = Tuple(Float(20), Float(20))
    grid_lattice_vector2 = Tuple(Float(20), Float(20))
    
    displacements = Array(dtype=np.float32, shape = (17, 17, 2))

    def _displacements_default(self):
        return (np.random.random((17, 17, 2)) - 0.5)*30
    
    trait_view = View(
        Item('visible'),
        Item('grid_origin'),
        Item('grid_lattice_vector1'),
        Item('grid_lattice_vector2'),
        )

    
class MyEventFilter(QtCore.QObject):
    def __init__(self, callback=None):
        QtCore.QObject.__init__(self)
        self.callback = callback
        
    def eventFilter(self, obj, event):
        r = self.callback(obj, event)
        return bool(r)
    
    
class _GLImageWithOverlaysEditor(Editor):
    scrollable = True

    overlays = List( Instance( OverlayTrait, () ))
    # inherited trait 'value = Image'

    event_filter = Python

    def init(self, parent):
        self.control = GLImgWithOverlaysWidget()
        for k, o in enumerate(self.overlays):
                self._init_overlayGL(o, idx=k)

        self.set_tooltip()

        self.event_filter = MyEventFilter(callback=self.handle_mouse_event) #need to keep reference
        self.control.installEventFilter(self.event_filter) #TODO: install event filter for interactivity
        
        #TODO:
        #unsync_value !?
        #def dispose(self):
        # print "cleanup editor"
        #self.sync_value(self.factory.overlays, 
        #                'overlays', 
        #                is_list=True) ###
        # Editor.dispose(self)

        self.on_trait_change(self.update_editor_overlays, 'overlays[]', dispatch='ui', deferred=True)
        self.on_trait_change(self.update_editor_overlay_item, 'overlays:+', dispatch='ui', deferred=True)
        self.sync_value(self.factory.overlays, 'overlays', 'from', is_list = True) #after on_trait_change!


    evt_dict = {
        QtCore.QEvent.MouseButtonPress: 'left_click',
        QtCore.QEvent.MouseMove: 'drag',
        QtCore.QEvent.MouseButtonRelease: 'release',
        }
     
    def handle_mouse_event(self, obj, event):
        #print "event", event.type()
        #MouseMove, MouseButtonRelease, MouseButtonDblClick, Wheel, #TouchBegin, TouchUpdate, TouchEnd

        msg = self.evt_dict.get(event.type())
        if msg is not None:

            pos = event.pos()
            xs,ys  = pos.x(), pos.y()
            x, y = self.control.screen_to_img(xs, ys)
            xp, yp = 0.5*(x+1)*self.value.w, 0.5*(y+1)*self.value.h #pixel coordinates of texture
            #print "coordinates", xp, yp
            
            for o in self.overlays:
                if o.visible:
                    try:
                        r = o.handle_mouse(xp, yp, msg)
                    except Exception as e:
                        print("ignoring exception in mouse event handler", e)
                        r = False
                    
                    if r:
                        return True
            
        return False

    def update_editor(self):
        #print 'update editor'
        self.control.set_camimage(self.value)
        self.control.do_update('update_editor')

    #note: dispatched in ui thread
    def update_editor_overlays(self, obj, name, old, new):
        if self.control is not None:
            for k, o in enumerate(self.overlays):
                self._init_overlayGL(o, idx=k)

            self.control.do_update('update_editor_overlays') #TODO: dangerous (not in ui thread?)
        else:
            print("ERROR, control not yet initialized")

    #NOTE: dispatched in ui thread
    def update_editor_overlay_item(self, obj, name, old, new):
        if self.control is not None:
            self._modify_overlay(obj)
            #TODO: removed temporarily: leave update of control to image updates
            self.control.do_update('update_editor_overlay_item')
        else:
            print('warning in update_editor_overlay_item: control is None (not yet initialized?)', self, obj, name)

    def _init_overlayGL(self, overlay, idx):
        #print "init_overlay", idx, overlay.__class__.__name__
        overlay.idx = idx
        if isinstance(overlay, OverlayTraitCircle):
            glovl = GLOverlayCircle()
        elif isinstance(overlay, OverlayTraitRectangle):
            glovl = GLOverlayRectangle()
        elif isinstance(overlay, OverlayTraitGrid):
            glovl = GLOverlayGrid()
        elif isinstance(overlay, OverlayTraitSpots):
            glovl = GLOverlaySpots()
        elif isinstance(overlay, OverlayTraitArrow):
            glovl = GLOverlayArrow()
        elif isinstance(overlay, OverlayTraitWarpArrows):
            glovl = GLOverlayWarpArrows()
        else:
            raise Exception('unknown overlay class')
        self.control.gloverlays[idx] = glovl
        self._modify_overlay(overlay)
        
    def _modify_overlay(self, overlay):
        "update control GLoverlay according to overlay trait" #called when overlay trait has been changed
        try:
            nx, ny = float(self.value.w), float(self.value.h) #TODO: self.value could be None!
        except AttributeError:
            nx, ny = 640, 480
            print('Exception caught in glimgwidget/_modify_overlay')
        glovl = self.control.gloverlays[overlay.idx]
        glovl.update_overlay(overlay_trait=overlay, img_size=(nx,ny)) #TODO: dangerous (not in ui thread?), possibly not, changes only trait attributes


class GLImageWithOverlaysEditor(BasicEditorFactory):
    klass = _GLImageWithOverlaysEditor
    overlays = Str #trait name of overlays

###############
# Demos / Tests
###############

def qApp():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        print('no qt app found, create one')
        app = QtGui.QApplication([])
    else:
        print('found qt app instance')  # likely, import traitsui initializes qApp

    return app

def start_event_loop():
    from IPython.lib.guisupport import start_event_loop_qt4, is_event_loop_running_qt4
    if not is_event_loop_running_qt4():
        print('qt4 event loop not running')
        print('starting event loop via IPython.lib.guisupport.start_event_loop_qt4')
        start_event_loop_qt4(QtCore.QCoreApplication.instance())
    else:
        print('running qt4 event loop detected')


class DemoWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        window = QtGui.QWidget()
        self.glimgwidget = GLImgWidget(parent=window)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.glimgwidget, 1, QtCore.Qt.AlignCenter)
        window.setLayout(layout)
        self.setWindowTitle('GLImgWidget demo')
        self.setCentralWidget(window)

def test_demo_window():
    window = DemoWindow()
    window.show()
    return window

def test_cl_texture():
    window = DemoWindow()
    window.show()
    g = window.glimgwidget
    g.update_texture_cl()
    return window, g


class DemoOverlayWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        window = QtGui.QWidget()
        self.imgwidget = GLImgWithOverlaysWidget(None)
        self.imgwidget.gloverlays[0] = GLOverlayCircle()
        self.imgwidget.gloverlays[1] = GLOverlayRectangle()
        self.imgwidget.gloverlays[2] = GLOverlayWarpArrows()
        
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.imgwidget, 1, QtCore.Qt.AlignCenter)
        window.setLayout(layout)
        self.setWindowTitle('OpenGL image widget')
        self.setCentralWidget(window)

def test_demo_overlay_window():
    window = DemoOverlayWindow()
    window.show()
    return window

def test_SLM_window():
    app = qApp()

    desktop = app.desktop()
    print("desktop screen count", desktop.screenCount())
    print("desktop is virtual desktop", desktop.isVirtualDesktop())
    print("screen geometry default screen", desktop.screenGeometry(screen=-1))
    print("primary screen index", desktop.primaryScreen())
    for k in range(desktop.screenCount()):
        print("screen geometry screen #%d: %s" % (k, desktop.screenGeometry(screen=k)))

    window = SLMWindow()

    print("window created")
    window.show()
    return window


class TraitsTestImage(HasTraits):
    image = Instance(CamImage, CamImage())

    traits_view = View(
        Item('image',
             editor = GLImageEditor(),
             ),
        resizable=True,
    )

def test_traits_image():
    t = TraitsTestImage()
    e = t.edit_traits()
    return t, e

class TraitsTestImageOverlays(HasTraits):
    image = Instance(CamImage)
    overlays = List(Instance(OverlayTrait), [OverlayTraitCircle(), 
                                             OverlayTraitRectangle(),
                                             OverlayTraitGrid(),
                                             OverlayTraitSpots(),
                                             OverlayTraitArrow(),
                                             OverlayTraitWarpArrows(),
                                             ])

    def _image_default(self):
        data = np.zeros((480, 640,), np.uint8) #rows, columns
        #data[::2, ::2] = 213
        #data[::4, ::4] = 50
        #data[:32:2, -32:-1:2] = 128
        data[200] = 255
        data[300] = 200
        data[:,256] = 255
        camimg = CamImage()
        camimg.data_roi = data
        return camimg
    
    traits_view = View(
        Item('image', editor = GLImageWithOverlaysEditor(overlays = 'overlays'),
             springy = True),
        Item('overlays',
             style = 'custom',
             editor = ListEditor(use_notebook = True),
             ),
        resizable=True,
             )

def test_traits_image_overlays():
    t = TraitsTestImageOverlays()
    e = t.edit_traits()
    return t, e
        

if __name__ == '__main__':

    #w = test_demo_window() #keeping reference important!
    #w = test_demo_overlay_window()  # keeping reference important!
    #w = test_SLM_window()

    #w, g = test_cl_texture()

    ## Test updating texture via cl kernel
    #w = test_demo_window()
    #g = w.glimgwidget
    #g.update_texture_cl()

    # Traits test
    #t, e = test_traits_image()
    t, e = test_traits_image_overlays()

    ## old stuff
    #glot = e._editors[0].control.gloverlays[5]
    #o = t.overlays[-1]

    start_event_loop()
