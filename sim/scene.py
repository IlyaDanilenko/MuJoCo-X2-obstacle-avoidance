# sim/scene.py - код генератора сцен

import mujoco
import os
from collections import Counter
import re

class SceneGenerator:
    def __init__(self, modelname="Skydio X2 scene"):
        """
        Инициализация генератора
        
        Args:
            modelname: Имя модели
        """
        self.original_dir = os.getcwd()
        
        # Определяем директорию со сценой
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        scene_xml_path = os.path.join(project_root, 'mujoco_menagerie-main', 'skydio_x2', 'scene.xml')
        if os.path.exists(scene_xml_path):
            scene_dir = os.path.dirname(scene_xml_path)
        else:
            potential_dir = os.path.join(project_root, 'mujoco_menagerie-main', 'skydio_x2')
            if os.path.exists(potential_dir):
                scene_dir = potential_dir
            else:
                raise FileNotFoundError()
        
        self.scene_dir = os.path.abspath(scene_dir)
        if not os.path.exists(self.scene_dir):
            raise FileNotFoundError(f"Директория сцены не существует: {self.scene_dir}")
        
        os.chdir(self.scene_dir)
        
        self.spec = mujoco.MjSpec()
        self.spec.modelname = modelname
        
        self.x2_default = None
        self.visual_default = None
        self.collision_default = None
        self.rotor_default = None
        
        # Счетчик столбов для автоматических имен
        self.pillar_counter = 0
        # Множество использованных имен для проверки уникальности
        self.used_names = set()
        
        # Флаг компиляции
        self.compiled = False
        self.model = None
        # Флаг генерации базовой сцены
        self.base_scene_generated = False
    
    def generate_base_scene(self):
        """Генерация базовой сцены с дроном"""
        if self.base_scene_generated:
            raise RuntimeError("Базовая сцена уже была сгенерирована! Нельзя вызывать generate_base_scene() дважды.")
        
        spec = self.spec
        
        # Настройки compiler
        spec.compiler.autolimits = True
        
        # Настройки option
        spec.option.timestep = 0.01
        spec.option.density = 1.225
        spec.option.viscosity = 1.8e-5
        
        # Настройки statistic
        spec.stat.center = [0, 0, 0.1]
        spec.stat.extent = 0.6
        spec.stat.meansize = 0.05
        
        # Настройки visual
        spec.visual.headlight.diffuse = [0.8, 0.8, 0.8]
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.specular = [0, 0, 0]
        spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
        spec.visual.global_.azimuth = -20
        spec.visual.global_.elevation = -20
        spec.visual.global_.ellipsoidinertia = True
        spec.visual.global_.offwidth = 640
        spec.visual.global_.offheight = 480
        
        # Создаем default классы
        main_default = spec.default
        self.x2_default = spec.add_default('x2', main_default)
        self.x2_default.geom.mass = 0
        self.x2_default.mesh.scale = [0.01, 0.01, 0.01]
        
        self.visual_default = spec.add_default('visual', self.x2_default)
        self.visual_default.geom.group = 2
        self.visual_default.geom.type = mujoco.mjtGeom.mjGEOM_MESH
        self.visual_default.geom.contype = 0
        self.visual_default.geom.conaffinity = 0
        
        self.collision_default = spec.add_default('collision', self.x2_default)
        self.collision_default.geom.group = 3
        self.collision_default.geom.type = mujoco.mjtGeom.mjGEOM_BOX
        
        self.rotor_default = spec.add_default('rotor', self.collision_default)
        self.rotor_default.geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        self.rotor_default.geom.size = [0.13, 0.13, 0.01]
        
        # Добавляем assets из scene.xml
        skybox_texture = spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                                          builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT)
        skybox_texture.rgb1 = [0.3, 0.5, 0.7]
        skybox_texture.rgb2 = [0, 0, 0]
        skybox_texture.width = 512
        skybox_texture.height = 3072
        
        groundplane_texture = spec.add_texture(name="groundplane", 
                                               type=mujoco.mjtTexture.mjTEXTURE_2D, 
                                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER)
        groundplane_texture.rgb1 = [0.2, 0.3, 0.4]
        groundplane_texture.rgb2 = [0.1, 0.2, 0.3]
        groundplane_texture.markrgb = [0.8, 0.8, 0.8]
        groundplane_texture.width = 300
        groundplane_texture.height = 300
        
        groundplane_material = spec.add_material(name="groundplane")
        groundplane_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
        groundplane_material.texuniform = True
        groundplane_material.texrepeat = [5, 5]
        groundplane_material.reflectance = 0.2
        
        # Добавляем assets из x2.xml
        x2_texture = spec.add_texture(type=mujoco.mjtTexture.mjTEXTURE_2D, 
                                      file="X2_lowpoly_texture_SpinningProps_1024.png")
        
        phong3SG_material = spec.add_material(name="phong3SG")
        phong3SG_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "X2_lowpoly_texture_SpinningProps_1024"
        
        invisible_material = spec.add_material(name="invisible")
        invisible_material.rgba = [0, 0, 0, 0]
        
        x2_mesh = spec.add_mesh(name="X2_lowpoly", file="X2_lowpoly.obj")
        x2_mesh.classname = self.x2_default
        x2_mesh.scale = [0.01, 0.01, 0.01]
        
        # Загружаем файлы assets в словарь
        spec.assets = {}
        assetdir = "assets"
        if os.path.exists(assetdir):
            for filename in os.listdir(assetdir):
                filepath = os.path.join(assetdir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, 'rb') as f:
                        file_content = f.read()
                        spec.assets[filename] = file_content
        
        # Добавляем worldbody элементы из scene.xml
        scene_light = spec.worldbody.add_light()
        scene_light.pos = [0, 0, 1.5]
        scene_light.dir = [0, 0, -1]
        
        floor_geom = spec.worldbody.add_geom(name="floor")
        floor_geom.size = [0, 0, 0.05]
        floor_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor_geom.material = "groundplane"
        self.used_names.add("floor")  # Добавляем имя пола в множество использованных
        
        # Добавляем worldbody элементы из x2.xml
        spotlight = spec.worldbody.add_light(name="spotlight")
        spotlight.pos = [0, -1, 2]
        self.used_names.add("spotlight")
        
        x2_body = spec.worldbody.add_body(name="x2", pos=[0, 0, 0.1])
        self.used_names.add("x2")
        x2_body.childclass = "x2"
        
        # Добавляем freejoint
        x2_body.add_freejoint()
        
        # Добавляем камеры
        track_cam = x2_body.add_camera(name="track", pos=[-1, 0, 0.5], xyaxes=[0, -1, 0, 1, 0, 2])
        drone_cam = x2_body.add_camera(name="drone_camera", pos=[-0.3, 0, 0], 
                                       xyaxes=[0, 1, 0, 0.1736, 0, 0.9848])
        drone_cam.fovy = 70
        
        # Добавляем sites
        imu_site = x2_body.add_site(name="imu", pos=[0, 0, 0.02])
        thrust1_site = x2_body.add_site(name="thrust1", pos=[-0.14, -0.18, 0.05])
        thrust2_site = x2_body.add_site(name="thrust2", pos=[-0.14, 0.18, 0.05])
        thrust3_site = x2_body.add_site(name="thrust3", pos=[0.14, 0.18, 0.08])
        thrust4_site = x2_body.add_site(name="thrust4", pos=[0.14, -0.18, 0.08])
        
        # Добавляем geoms
        visual_geom = x2_body.add_geom()
        visual_geom.classname = self.visual_default
        visual_geom.type = mujoco.mjtGeom.mjGEOM_MESH
        visual_geom.group = 2
        visual_geom.material = "phong3SG"
        visual_geom.meshname = "X2_lowpoly"
        visual_geom.quat = [0, 0, 1, 1]
        
        collision1 = x2_body.add_geom()
        collision1.classname = self.collision_default
        collision1.type = mujoco.mjtGeom.mjGEOM_BOX
        collision1.group = 3
        collision1.size = [0.06, 0.027, 0.02]
        collision1.pos = [0.04, 0, 0.02]
        
        collision2 = x2_body.add_geom()
        collision2.classname = self.collision_default
        collision2.type = mujoco.mjtGeom.mjGEOM_BOX
        collision2.group = 3
        collision2.size = [0.06, 0.027, 0.02]
        collision2.pos = [0.04, 0, 0.06]
        
        collision3 = x2_body.add_geom()
        collision3.classname = self.collision_default
        collision3.type = mujoco.mjtGeom.mjGEOM_BOX
        collision3.group = 3
        collision3.size = [0.05, 0.027, 0.02]
        collision3.pos = [-0.07, 0, 0.065]
        
        collision4 = x2_body.add_geom()
        collision4.classname = self.collision_default
        collision4.type = mujoco.mjtGeom.mjGEOM_BOX
        collision4.group = 3
        collision4.size = [0.023, 0.017, 0.01]
        collision4.pos = [-0.137, 0.008, 0.065]
        collision4.quat = [1, 0, 0, 1]
        
        rotor1 = x2_body.add_geom(name="rotor1")
        rotor1.classname = self.rotor_default
        rotor1.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        rotor1.size = [0.13, 0.13, 0.01]
        rotor1.group = 3
        rotor1.pos = [-0.14, -0.18, 0.05]
        rotor1.mass = 0.25
        self.used_names.add("rotor1")
        
        rotor2 = x2_body.add_geom(name="rotor2")
        rotor2.classname = self.rotor_default
        rotor2.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        rotor2.size = [0.13, 0.13, 0.01]
        rotor2.group = 3
        rotor2.pos = [-0.14, 0.18, 0.05]
        rotor2.mass = 0.25
        self.used_names.add("rotor2")
        
        rotor3 = x2_body.add_geom(name="rotor3")
        rotor3.classname = self.rotor_default
        rotor3.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        rotor3.size = [0.13, 0.13, 0.01]
        rotor3.group = 3
        rotor3.pos = [0.14, 0.18, 0.08]
        rotor3.mass = 0.25
        self.used_names.add("rotor3")
        
        rotor4 = x2_body.add_geom(name="rotor4")
        rotor4.classname = self.rotor_default
        rotor4.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        rotor4.size = [0.13, 0.13, 0.01]
        rotor4.group = 3
        rotor4.pos = [0.14, -0.18, 0.08]
        rotor4.mass = 0.25
        self.used_names.add("rotor4")
        
        invisible_geom = x2_body.add_geom()
        invisible_geom.size = [0.16, 0.04, 0.02]
        invisible_geom.pos = [0, 0, 0.02]
        invisible_geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        invisible_geom.mass = 0.325
        invisible_geom.classname = self.visual_default
        invisible_geom.material = "invisible"
        
        # Добавляем actuators
        motor1 = spec.add_actuator(name="thrust1", target="thrust1", trntype=mujoco.mjtTrn.mjTRN_SITE)
        motor1.classname = self.x2_default
        motor1.gear = [0, 0, 1, 0, 0, 0.0201]
        motor1.ctrlrange = [0, 13]
        
        motor2 = spec.add_actuator(name="thrust2", target="thrust2", trntype=mujoco.mjtTrn.mjTRN_SITE)
        motor2.classname = self.x2_default
        motor2.gear = [0, 0, 1, 0, 0, -0.0201]
        motor2.ctrlrange = [0, 13]
        
        motor3 = spec.add_actuator(name="thrust3", target="thrust3", trntype=mujoco.mjtTrn.mjTRN_SITE)
        motor3.classname = self.x2_default
        motor3.gear = [0, 0, 1, 0, 0, 0.0201]
        motor3.ctrlrange = [0, 13]
        
        motor4 = spec.add_actuator(name="thrust4", target="thrust4", trntype=mujoco.mjtTrn.mjTRN_SITE)
        motor4.classname = self.x2_default
        motor4.gear = [0, 0, 1, 0, 0, -0.0201]
        motor4.ctrlrange = [0, 13]
        
        # Добавляем sensors
        gyro = spec.add_sensor(name="body_gyro", type=mujoco.mjtSensor.mjSENS_GYRO)
        gyro.objtype = mujoco.mjtObj.mjOBJ_SITE
        gyro.objname = "imu"
        
        accel = spec.add_sensor(name="body_linacc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER)
        accel.objtype = mujoco.mjtObj.mjOBJ_SITE
        accel.objname = "imu"
        
        framequat = spec.add_sensor(name="body_quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT)
        framequat.objtype = mujoco.mjtObj.mjOBJ_SITE
        framequat.objname = "imu"
        
        # Добавляем keyframe
        hover_key = spec.add_key(name="hover")
        hover_key.qpos = [0, 0, 0.3, 1, 0, 0, 0]
        hover_key.ctrl = [3.2495625, 3.2495625, 3.2495625, 3.2495625]
        
        self.base_scene_generated = True
    
    def add_pillar(self, x, y, radius, height, name=None, rgba=None):
        """
        Добавление столба (цилиндра) в сцену
        
        Args:
            x: X координата центра столба
            y: Y координата центра столба
            radius: Радиус столба
            height: Высота столба
            name: Имя столба (если None, генерируется автоматически)
            rgba: Цвет RGBA [r, g, b, a] (опционально, если не указан material)
        
        Returns:
            Созданный geom объект
        """
        if name is None:
            self.pillar_counter += 1
            base_name = f"pillar_{self.pillar_counter}"
            name = base_name
            counter = 1
            while name in self.used_names:
                name = f"{base_name}_{counter}"
                counter += 1
        else:
            base_name = name
            counter = 1
            original_name = name
            while name in self.used_names:
                name = f"{base_name}_{counter}"
                counter += 1
            if name != original_name:
                print(f"⚠ Имя '{original_name}' уже используется, переименовано в '{name}'")
        
        if name in self.used_names:
            raise ValueError(f"Критическая ошибка: имя '{name}' уже используется в used_names, но должно было быть переименовано!")
        self.used_names.add(name)
        
        pos = [x, y, height / 2]
        
        size = [radius, height / 2, height / 2]
        
        # Создаем geom в worldbody
        try:
            pillar_geom = self.spec.worldbody.add_geom(name=name)
        except Exception as e:
            # Если произошла ошибка при создании, удаляем имя из used_names
            self.used_names.discard(name)
            raise ValueError(f"Ошибка при создании столба с именем '{name}': {e}")
        
        pillar_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        pillar_geom.size = size
        pillar_geom.pos = pos
        pillar_geom.group = 0
        pillar_geom.contype = 1
        pillar_geom.conaffinity = 1
        
        if rgba:
            pillar_geom.rgba = rgba
        else:
            # Цвет по умолчанию - серый
            pillar_geom.rgba = [0.7, 0.7, 0.7, 1]
        
        return pillar_geom
    
    def compile(self):
        """
        Компиляция модели
        
        Returns:
            Скомпилированная модель (mujoco.MjModel)
        """
        if self.compiled:
            return self.model
        
        # Проверяем XML перед компиляцией на дубликаты имен
        xml_content = self.spec.to_xml()
        geom_names = re.findall(r'<geom[^>]*name="([^"]+)"', xml_content)
        
        name_counts = Counter(geom_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if duplicates:
            print(f"⚠ Найдены дубликаты имен geoms в XML: {duplicates}")
            print("Используемые имена:", self.used_names)
            raise
        
        self.model = self.spec.compile()
        self.compiled = True
        return self.model
    
    def __enter__(self):
        """Поддержка контекстного менеджера"""
        return self

class Scene:
  """Класс для управления сценой (окружением)"""
  def __init__(self, scene_path=None):
    """
    Инициализация сцены
    
    Args:
        scene_path: Путь к XML файлу сцены. Если None, создается пустая сцена через SceneGenerator
    """
    if scene_path:
      self.m = mujoco.MjModel.from_xml_path(scene_path)
      self.d = mujoco.MjData(self.m)
      self.generator = None
    else:
      self.generator = SceneGenerator()
      self.generator.generate_base_scene()
      self.m = self.generator.compile()
      self.d = mujoco.MjData(self.m)
  
  def add_pillar(self, x, y, radius, height, name=None, rgba=None):
    """
    Добавляет столб в сцену через SceneGenerator
    
    Args:
        x: X координата центра столба
        y: Y координата центра столба
        radius: Радиус столба
        height: Высота столба
        name: Имя столба (опционально)
        rgba: Цвет RGBA [r, g, b, a] (опционально)
    """
    if self.generator is None:
      raise ValueError("Нельзя добавлять столбы в сцену, загруженную из файла. Создайте сцену через SceneGenerator.")
    
    self.generator.add_pillar(x=x, y=y, radius=radius, height=height, 
                              name=name, rgba=rgba)
    
    self.generator.compiled = False
    
    self.m = self.generator.compile()
    
    self.d = mujoco.MjData(self.m)
