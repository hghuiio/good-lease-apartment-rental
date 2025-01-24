package com.atguigu.lease.web.admin.service;

import com.atguigu.lease.model.entity.SystemUser;
import com.atguigu.lease.web.admin.mapper.SystemUserMapper;
import com.atguigu.lease.web.admin.vo.login.CaptchaVo;
import com.atguigu.lease.web.admin.vo.login.LoginVo;
import com.atguigu.lease.web.admin.vo.system.user.SystemUserInfoVo;
import org.springframework.beans.factory.annotation.Autowired;

public interface LoginService {
    
  
    CaptchaVo getCaptcha();

    String login(LoginVo loginVo);


    SystemUserInfoVo getLoginUserInfo(Long userId);
}
