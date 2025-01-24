package com.atguigu.lease.common.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Random;

public class CodeUtil {
     @NotNull
     public static String getCode(int length) {
        StringBuilder builder = new StringBuilder();
        Random random = new Random();
        for (int i = 0; i < length; i++) {
            builder.append(random.nextInt(10));
        }
        return builder.toString();
    }


}
