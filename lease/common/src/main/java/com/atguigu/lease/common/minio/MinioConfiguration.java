package com.atguigu.lease.common.minio;

import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
@ConfigurationPropertiesScan("com.atguigu.lease.common.minio")
@Configuration
@ConditionalOnProperty(name = "minio.endpoint")
public class MinioConfiguration {
    @Autowired
  private MinioProperties Properties;

    @Bean
    public MinioClient minioClient(){

   return   MinioClient.builder().endpoint(Properties.getEndpoint()).credentials(Properties.getAccessKey(),Properties.getSecretKey()).build();
      }

}
