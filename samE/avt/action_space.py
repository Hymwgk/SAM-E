import torch
import numpy as np  
from math import sin,cos,pi,acos,atan2
from scipy.spatial.transform import Rotation as R




class SpherCartTransform():
    """关注世界坐标系和局部球坐标之间的转换关系"""
    
    def __init__(self,spher_center=None,
                 local_rotation=None,
                 boundary_offset=True) -> None:
        
        self.wtl_pos = np.array(spher_center) if spher_center else np.zeros([3,])
        self.wtl_rot = R.from_quat(local_rotation).as_matrix() if local_rotation else np.eye(3)
        
        self.boundary_offset = boundary_offset
    
    def set_spher_center(self,spher_center):
        """设置局部球坐标系的中心"""
        self.wtl_pos = np.array(spher_center)
    
    def set_local_rotation(self,local_rotation):
        """设置local坐标系的旋转"""
        self.wtl_rot = R.from_quat(local_rotation).as_matrix()
            
    def _cart_to_spher(self,cartesion_position:np.ndarray,):
        """输出弧度制"""
        # 目标点在local坐标系中的三维坐标
        ltp_trans = cartesion_position
        # 将ltc_pos笛卡尔坐标 转换为  local球坐标值
        local_r = np.linalg.norm(ltp_trans)
        local_theta = acos(ltp_trans[2]/local_r) 
        local_phi = atan2(ltp_trans[1],ltp_trans[0])
        
        if self.boundary_offset:
            if local_phi>np.pi:
                local_phi -= 2*np.pi

        #返回球坐标
        return np.array([local_r,local_theta,local_phi])


    def _spher_to_cart(self,local_spher_pos:np.ndarray,):
        """输入球坐标为弧度制
        """
        local_r,local_theta,local_phi = tuple(local_spher_pos)

        # local_theta = np.radians(local_theta)
        # local_phi = np.radians(local_phi)
        
        if self.boundary_offset:
            # 将phi变回规范的[0,2pi]范围 
            if local_phi<0:
                local_phi += 2*np.pi
            elif local_phi > 2*np.pi:
                local_phi -= 2*np.pi 
        
        # 计算local笛卡尔坐标
        local_pos_x = local_r* sin(local_theta)*cos(local_phi)
        local_pos_y = local_r* sin(local_theta)*sin(local_phi)
        local_pos_z = local_r* cos(local_theta)
        ltp_trans = np.array([local_pos_x,local_pos_y,local_pos_z])
        
        return ltp_trans
    
    def get_world_cart(self,local_spher_pos:np.ndarray):
        """局部球坐标转世界坐标"""
        ltp_trans = self._spher_to_cart(local_spher_pos)
        # 计算wtp
        wtp_trans = self.wtl_rot.dot(ltp_trans) + self.wtl_pos
        # 
        return wtp_trans
        

    def get_local_spher(self,world_cart_pos:np.ndarray):
        """将世界坐标系中的笛卡尔坐标，转换到局部球坐标系中的球坐标"""
        wtp_trans = world_cart_pos
        # 目标点在 local 视点笛卡尔坐标系下位置
        ltp_trans = self.wtl_rot.T.dot(wtp_trans) - self.wtl_rot.T.dot(self.wtl_pos)
        local_spher = self._cart_to_spher(ltp_trans)
        return local_spher

class ViewpointSpace():
    def __init__(self,
                 spher_bound,
                 cart_bound,
                 spher_center=None,
                 local_rotation=None,
                 boundary_offset=True) -> None:
        
        self.coord_tsf = SpherCartTransform(spher_center=spher_center,
                                            local_rotation=local_rotation,
                                            boundary_offset=boundary_offset)
        self.spher_bound = spher_bound
        self.cart_bound = cart_bound

    def boundary_check(self,boundary):
        pass
        
    def get_cart_pos(self,spher_viewpoint):
        """获取视点的世界笛卡尔坐标"""
        cart_pos = self.coord_tsf.get_world_cart(local_spher_pos=spher_viewpoint)
        #cart_pos = np.clip(cart_pos,self.cart_bound[:3],self.cart_bound[3:])
    
        return np.clip(cart_pos,self.cart_bound[:3],self.cart_bound[3:])
    
    def get_spher_pos(self,cart_viewpoint):
        """视点的世界笛卡尔坐标，转换为局部球坐标"""
        spher_coord = self.coord_tsf.get_local_spher(world_cart_pos=cart_viewpoint)
        return np.clip(spher_coord,self.spher_bound[:3],self.spher_bound[3:])
    
    