#!/usr/bin/env python3
"""Генератор вертикальных хоррор-роликов (1080x1920) в мрачной плоской 2D-стилистике.

    python3 build.py stories/derevnya_bez_ognej.json -o out.mp4

Конвейер: озвучка edge-tts с таймингами слов -> сжатие длинных пауз ->
саунд-дизайн (numpy) -> покадровая отрисовка сцен (cairo) -> сборка ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cairo
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
W, H, FPS, SR = 1080, 1920, 30, 48000
FONT = "Inter"
RNG = np.random.default_rng(7)

# ------------------------------------------------------------------ palette
def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)

OUT = hexc("#0c0d0f")
WALL = hexc("#3b4046")
WALL_D = hexc("#2c3035")
ROOF = hexc("#30343a")
WIN_D = hexc("#14161a")
WIN_L = hexc("#e9e0c8")
SKIN = hexc("#d6cfc4")
RED = hexc("#e0393e")


# ------------------------------------------------------------------ helpers
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def smooth_noise(t, seed=0.0, speed=1.0):
    """Гладкий псевдослучайный сигнал в [-1, 1]."""
    x = t * speed + seed * 17.13
    return (math.sin(x * 1.7) * 0.5 + math.sin(x * 2.9 + 1.3) * 0.3 + math.sin(x * 5.3 + 2.1) * 0.2)


def rrect(c, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    c.close_path()


def fill_stroke(c, fill, stroke=OUT, lw=5):
    c.set_source_rgba(*fill)
    c.fill_preserve()
    c.set_source_rgba(*stroke)
    c.set_line_width(lw)
    c.stroke()


def lin(c, x0, y0, x1, y1, stops):
    g = cairo.LinearGradient(x0, y0, x1, y1)
    for off, col in stops:
        g.add_color_stop_rgba(off, *col)
    return g


def rad(c, cx, cy, r0, r1, stops, sx=1.0, sy=1.0):
    g = cairo.RadialGradient(cx / sx, cy / sy, r0, cx / sx, cy / sy, r1)
    for off, col in stops:
        g.add_color_stop_rgba(off, *col)
    return g


def glow(c, cx, cy, r, col, a, sy=1.0):
    c.save()
    c.scale(1, sy)
    g = cairo.RadialGradient(cx, cy / sy, 0, cx, cy / sy, r)
    g.add_color_stop_rgba(0, col[0], col[1], col[2], a)
    g.add_color_stop_rgba(0.45, col[0], col[1], col[2], a * 0.35)
    g.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
    c.set_source(g)
    c.arc(cx, cy / sy, r, 0, 2 * math.pi)
    c.fill()
    c.restore()


def bg(c, top, bottom):
    c.set_source(lin(c, 0, 0, 0, H, [(0, hexc(top)), (1, hexc(bottom))]))
    c.paint()


# ------------------------------------------------------------------ characters
def hero(c, x, feet, s=1.0, look=0.0, flashlight=True):
    """Парень в шапке с помпоном и пуховике (вид спереди). Возвращает точку фонарика."""
    c.save()
    c.translate(x, feet)
    c.scale(s, s)
    # тень
    c.set_source(rad(c, 0, 0, 0, 120, [(0, (0, 0, 0, 0.55)), (1, (0, 0, 0, 0))]))
    c.save(); c.scale(1, 0.18); c.arc(0, 0, 130, 0, 2 * math.pi); c.fill(); c.restore()
    # ноги
    for sx in (-1, 1):
        rrect(c, sx * 34 - 22, -190, 44, 170, 12)
        fill_stroke(c, hexc("#2a2c30"))
        rrect(c, sx * 38 - 32, -34, 64, 34, 14)
        fill_stroke(c, hexc("#d4d2cc"))
    # левая рука (опущена)
    rrect(c, -118, -420, 46, 210, 22)
    fill_stroke(c, hexc("#474c52"))
    c.arc(-95, -205, 20, 0, 2 * math.pi)
    fill_stroke(c, SKIN, lw=4)
    # куртка
    rrect(c, -92, -440, 184, 270, 34)
    fill_stroke(c, hexc("#4c5157"))
    c.set_source_rgba(*hexc("#363a3f"))
    c.set_line_width(4)
    for yy in (-395, -345, -295, -245):
        c.move_to(-86, yy); c.curve_to(-30, yy + 9, 30, yy + 9, 86, yy); c.stroke()
    c.move_to(0, -440); c.line_to(0, -172); c.stroke()
    # воротник
    rrect(c, -58, -462, 116, 46, 18)
    fill_stroke(c, hexc("#43474d"))
    # правая рука с фонариком, вытянута вперёд-вправо
    c.save()
    c.translate(84, -405)
    c.rotate(-0.95)
    rrect(c, -22, -10, 46, 175, 22)
    fill_stroke(c, hexc("#474c52"))
    c.arc(0, 172, 21, 0, 2 * math.pi)
    fill_stroke(c, SKIN, lw=4)
    if flashlight:
        rrect(c, -12, 175, 24, 62, 6)
        fill_stroke(c, hexc("#22252a"), lw=4)
    c.restore()
    fx = 84 + math.sin(0.95) * 236
    fy = -405 + math.cos(0.95) * 236
    # голова
    c.arc(0, -545, 82, 0, 2 * math.pi)
    fill_stroke(c, SKIN)
    # глаза
    ex = look * 10
    c.set_source_rgba(*hexc("#16171a"))
    for sx in (-1, 1):
        c.arc(sx * 26 + ex, -538, 7.5, 0, 2 * math.pi); c.fill()
    c.set_line_width(4)
    c.move_to(-14 + ex, -497); c.line_to(14 + ex, -497); c.stroke()
    # шапка
    c.move_to(-88, -560)
    c.curve_to(-92, -660, 92, -660, 88, -560)
    c.close_path()
    fill_stroke(c, hexc("#6f645b"))
    rrect(c, -94, -585, 188, 44, 14)
    fill_stroke(c, hexc("#62574f"))
    c.set_source_rgba(*hexc("#4d443e"))
    c.set_line_width(3)
    for i in range(-80, 90, 16):
        c.move_to(i, -581); c.line_to(i, -545); c.stroke()
    c.arc(0, -655, 22, 0, 2 * math.pi)
    fill_stroke(c, hexc("#cfc8bd"))
    c.restore()
    return x + fx * s, feet + fy * s


def house(c, x, base, w, h, roof_h, wall=WALL, lit=0.0, door_open=0.0, win2=True, figure=0.0, detail=True):
    """Дом как в референсе: стена, треугольная крыша, окна, дверь."""
    top = base - h
    # крыша
    c.move_to(x - w / 2 - 28, top + 4)
    c.line_to(x, top - roof_h)
    c.line_to(x + w / 2 + 28, top + 4)
    c.close_path()
    fill_stroke(c, ROOF if wall == WALL else tuple(v * 0.8 for v in wall[:3]) + (1,))
    # стена
    c.rectangle(x - w / 2, top, w, h)
    fill_stroke(c, wall)
    if detail:  # доски
        c.set_source_rgba(0, 0, 0, 0.18)
        c.set_line_width(2)
        yy = top + 28
        while yy < base - 4:
            c.move_to(x - w / 2 + 3, yy); c.line_to(x + w / 2 - 3, yy); c.stroke(); yy += 28
    # окно 1
    ww, wh = w * 0.2, h * 0.3
    wx, wy = x - w * 0.36, top + h * 0.22
    c.rectangle(wx, wy, ww, wh)
    if lit > 0:
        col = tuple(WIN_D[i] + (WIN_L[i] - WIN_D[i]) * lit for i in range(3)) + (1,)
        fill_stroke(c, col)
        if figure > 0:  # силуэт в окне: длинный, худой, голова касается рамы
            c.save()
            c.rectangle(wx, wy, ww, wh); c.clip()
            c.set_source_rgba(0.05, 0.05, 0.06, figure)
            cx = wx + ww * 0.55
            c.arc(cx, wy + wh * 0.16, ww * 0.13, 0, 2 * math.pi); c.fill()
            c.move_to(cx - ww * 0.2, wy + wh * 1.0)
            c.curve_to(cx - ww * 0.2, wy + wh * 0.45, cx - ww * 0.1, wy + wh * 0.28, cx, wy + wh * 0.28)
            c.curve_to(cx + ww * 0.1, wy + wh * 0.28, cx + ww * 0.2, wy + wh * 0.45, cx + ww * 0.2, wy + wh)
            c.fill()
            c.restore()
        c.rectangle(wx, wy, ww, wh)
        c.set_source_rgba(*OUT); c.set_line_width(5); c.stroke()
        # конус света из окна на землю
        c.move_to(wx + 4, wy + wh)
        c.line_to(wx + ww - 4, wy + wh)
        c.line_to(wx + ww + w * 0.18, base + h * 0.42)
        c.line_to(wx - w * 0.2, base + h * 0.42)
        c.close_path()
        c.set_source(lin(c, 0, wy + wh, 0, base + h * 0.42,
                         [(0, (0.93, 0.89, 0.8, 0.45 * lit)), (1, (0.93, 0.89, 0.8, 0))]))
        c.fill()
        glow(c, wx + ww / 2, wy + wh / 2, ww * 1.6, (0.95, 0.9, 0.78), 0.35 * lit)
    else:
        fill_stroke(c, WIN_D)
        c.set_source_rgba(*OUT); c.set_line_width(4)
        c.move_to(wx + ww / 2, wy); c.line_to(wx + ww / 2, wy + wh); c.stroke()
        c.move_to(wx, wy + wh / 2); c.line_to(wx + ww, wy + wh / 2); c.stroke()
    if win2:
        c.rectangle(x + w * 0.14, wy, ww * 1.05, wh)
        fill_stroke(c, WIN_D)
    # дверь
    dw, dh = w * 0.17, h * 0.5
    dx = x - w * 0.12
    c.rectangle(dx, base - dh, dw, dh)
    fill_stroke(c, hexc("#08090a") if door_open > 0 else hexc("#4a4e54"))
    if door_open > 0:
        c.rectangle(dx, base - dh, dw * (1 - door_open), dh)
        fill_stroke(c, hexc("#4a4e54"))


# ------------------------------------------------------------------ scenes
class Ctx:
    def __init__(self, ev):
        self.ev = ev  # абсолютные времена событий


def forest_layers(c, t, drift, fog_a=0.35, seed=1):
    rng = np.random.default_rng(seed)
    for layer, (col, n, wmin, wmax, par) in enumerate([
        ("#2a2d31", 22, 14, 26, 0.3), ("#1f2124", 14, 26, 48, 0.6), ("#121315", 7, 46, 90, 1.0)]):
        xs = rng.uniform(-200, W + 200, n)
        ws = rng.uniform(wmin, wmax, n)
        for xx, ww in zip(xs, ws):
            xx = xx + drift * par
            c.rectangle(xx, -50, ww, 1300 + layer * 40)
            c.set_source_rgba(*hexc(col))
            c.fill()
            if layer == 2:  # ветки
                c.set_source_rgba(*hexc(col)); c.set_line_width(7)
                c.move_to(xx + ww / 2, 420 + (xx % 200)); c.line_to(xx + ww / 2 + 110, 330 + (xx % 200)); c.stroke()
        # туман между слоями
        c.set_source(lin(c, 0, 900, 0, 1300, [(0, (0.6, 0.62, 0.65, 0)), (0.6, (0.6, 0.62, 0.65, fog_a * 0.25)),
                                                (1, (0.6, 0.62, 0.65, 0))]))
        c.rectangle(0, 900, W, 400); c.fill()


def ground(c, y, top="#191a1d", bottom="#0d0e10"):
    c.set_source(lin(c, 0, y, 0, H, [(0, hexc(top)), (1, hexc(bottom))]))
    c.rectangle(-200, y, W + 400, H - y + 200)
    c.fill()
    c.set_source_rgba(1, 1, 1, 0.04)
    c.set_line_width(3)
    c.move_to(-200, y); c.line_to(W + 200, y); c.stroke()


def beam(c, x, y, x1, y1, spread, a):
    """Луч фонарика конусом."""
    if a <= 0.01:
        return
    ang = math.atan2(y1 - y, x1 - x)
    L = math.hypot(x1 - x, y1 - y)
    nx, ny = -math.sin(ang), math.cos(ang)
    c.move_to(x + nx * 6, y + ny * 6)
    c.line_to(x1 + nx * spread, y1 + ny * spread)
    c.line_to(x1 - nx * spread, y1 - ny * spread)
    c.line_to(x - nx * 6, y - ny * 6)
    c.close_path()
    c.set_source(lin(c, x, y, x + math.cos(ang) * L, y + math.sin(ang) * L,
                     [(0, (0.95, 0.93, 0.85, 0.42 * a)), (1, (0.95, 0.93, 0.85, 0.0))]))
    c.fill()
    glow(c, x1, y1, spread * 1.4, (0.9, 0.88, 0.8), 0.22 * a, sy=0.35)


def flashlight_level(t, ev):
    """Яркость фонарика: мигает на «сел», дальше слабый, гаснет на щелчке."""
    lvl = 1.0
    if t > ev["flicker"] - 0.2:
        k = t - ev["flicker"] + 0.2
        if k < 1.4:
            lvl = 0.15 if (math.sin(k * 47) + math.sin(k * 23.0)) > 0.6 else 0.9
        else:
            lvl = 0.55 + 0.08 * smooth_noise(t, 3, 9)
    if t >= ev["click"]:
        lvl = 0.0
    return lvl


def sc_forest(c, t, lt, T, ev):
    bg(c, "#1c1e21", "#101113")
    forest_layers(c, t, drift=-lt * 14)
    ground(c, 1220)
    fl = flashlight_level(t, ev)
    walk = math.sin(lt * 5.2) * 6
    hx, hy, hs = 440, 1330, 0.95
    fx, fy = hx + (84 + math.sin(0.95) * 236) * hs, hy + walk + (-405 + math.cos(0.95) * 236) * hs
    c.set_operator(cairo.OPERATOR_ADD)
    beam(c, fx, fy, 1060, 1290 + smooth_noise(t, 5, 2) * 20, 190, fl * 1.4)
    c.set_operator(cairo.OPERATOR_OVER)
    hero(c, hx, hy + walk, hs, look=0.4)


def sc_village(c, t, lt, T, ev):
    bg(c, "#17191c", "#0f1012")
    pan = -lt * 10
    # луна
    glow(c, 860 + pan * 0.2, 330, 230, (0.75, 0.75, 0.72), 0.25)
    c.arc(860 + pan * 0.2, 330, 64, 0, 2 * math.pi)
    c.set_source_rgba(*hexc("#7d7d78")); c.fill()
    # дальний лес
    forest_layers(c, t, drift=pan * 0.3, fog_a=0.5, seed=5)
    ground(c, 1240)
    # дальние дома
    for i, (xx, sc) in enumerate([(120, 0.45), (330, 0.55), (930, 0.5), (1120, 0.4)]):
        house(c, xx + pan * 0.6, 1250, 300 * sc * 2, 260 * sc * 2, 120 * sc * 2,
              wall=hexc("#2a2e33"), detail=False, win2=i % 2 == 0)
    # главный дом
    house(c, 560 + pan, 1360, 470, 340, 170)
    # забор
    c.set_source_rgba(*hexc("#1a1c1f"))
    for i in range(-2, 16):
        xx = i * 70 + pan * 1.3
        c.rectangle(xx, 1300, 16, 110); c.fill()
    c.rectangle(-100, 1325, W + 200, 10); c.fill()
    # карточка «0 огней»
    if t >= ev["card"]:
        k = ease((t - ev["card"]) / 0.35)
        c.set_source_rgba(0, 0, 0, 0.55 * k); c.paint()
        card(c, k, "ДЕРЕВНЯ", ["9 ДОМОВ", "0 ОГНЕЙ"])


def card(c, k, tag, lines):
    c.save()
    c.translate(W / 2, 640 - (1 - k) * 80)
    c.rotate(-0.025 - (1 - k) * 0.08)
    c.set_source_rgba(0, 0, 0, 0.5 * k)
    rrect(c, -425 + 10, -170 + 14, 850, 340, 8); c.fill()
    rrect(c, -425, -170, 850, 340, 8)
    c.set_source(lin(c, 0, -170, 0, 170, [(0, hexc("#ebe8e1", k)), (1, hexc("#d8d4cb", k))]))
    c.fill()
    # скотч
    c.rectangle(-55, -195, 110, 50)
    c.set_source_rgba(0.75, 0.75, 0.72, 0.8 * k); c.fill()
    # плашка
    c.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(30)
    ext = c.text_extents(tag)
    rrect(c, -ext.width / 2 - 34, -120, ext.width + 68, 54, 27)
    c.set_source_rgba(*hexc("#a8484b", k)); c.fill()
    c.set_source_rgba(1, 1, 1, k)
    c.move_to(-ext.width / 2 - ext.x_bearing, -93 - ext.y_bearing - ext.height / 2); c.show_text(tag)
    c.select_font_face("Inter ExtraBold", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(76)
    for i, ln in enumerate(lines):
        ext = c.text_extents(ln)
        c.move_to(-ext.width / 2 - ext.x_bearing, -10 + i * 88 + 30)
        c.set_source_rgba(*hexc("#26272a", k)); c.show_text(ln)
    c.restore()


def sc_door(c, t, lt, T, ev):
    bg(c, "#1e2024", "#121315")
    # стена из досок
    for i in range(0, W + 60, 64):
        c.rectangle(i, 0, 60, H)
        c.set_source_rgba(*hexc("#2b2f34" if (i // 64) % 2 else "#292c31")); c.fill()
        c.set_source_rgba(0, 0, 0, 0.35); c.set_line_width(4)
        c.move_to(i, 0); c.line_to(i, H); c.stroke()
    # дверной проём
    dx, dy, dw, dh = 300, 420, 480, 1060
    c.rectangle(dx - 26, dy - 26, dw + 52, dh + 26)
    fill_stroke(c, hexc("#1b1d20"))
    c.rectangle(dx, dy, dw, dh)
    c.set_source_rgba(0.01, 0.01, 0.012, 1); c.fill()
    # дверь приоткрывается внутрь
    op = 0.18 + 0.32 * ease(lt / max(T, 0.1))
    c.move_to(dx, dy); c.line_to(dx + dw * (1 - op), dy + 30 * op); c.line_to(dx + dw * (1 - op), dy + dh - 30 * op)
    c.line_to(dx, dy + dh); c.close_path()
    fill_stroke(c, hexc("#3d4146"))
    c.arc(dx + dw * (1 - op) - 40, dy + dh * 0.52, 12, 0, 2 * math.pi)
    fill_stroke(c, hexc("#77746e"), lw=3)
    # порог и снег
    ground(c, dy + dh, "#24262a", "#101113")
    # пятно фонарика
    fl = flashlight_level(t, ev)
    sx = 560 + smooth_noise(t, 2, 1.3) * 60
    sy = 980 + smooth_noise(t, 4, 1.1) * 50
    c.set_operator(cairo.OPERATOR_ADD)
    glow(c, sx, sy, 420, (0.55, 0.53, 0.48), 0.55 * fl)
    c.set_operator(cairo.OPERATOR_OVER)
    # тьма в щели «смотрит»
    if lt > T * 0.6:
        a = ease((lt - T * 0.6) / (T * 0.4)) * 0.8
        for sxx in (-14, 14):
            c.arc(dx + dw * (1 - op) + 70 + sxx, dy + 520, 3.5, 0, 2 * math.pi)
            c.set_source_rgba(0.85, 0.85, 0.8, a); c.fill()


def person_back(c, x, feet, s, scarf=False, turn=0.0):
    """Человек спиной к нам — тёмный силуэт."""
    c.save(); c.translate(x, feet); c.scale(s, s)
    col = hexc("#0e0f11")
    c.move_to(-110, 0)
    c.curve_to(-120, -260, -110, -440, -70, -470)
    c.line_to(70, -470)
    c.curve_to(110, -440, 120, -260, 110, 0)
    c.close_path()
    c.set_source_rgba(*col); c.fill_preserve()
    c.set_source_rgba(0.45, 0.52, 0.6, 0.35); c.set_line_width(4); c.stroke()
    if scarf:
        c.move_to(-82, -470); c.curve_to(-90, -640, 90, -640, 82, -470); c.close_path()
        c.set_source_rgba(*hexc("#141518")); c.fill_preserve()
    else:
        c.arc(0, -540, 72, 0, 2 * math.pi); c.set_source_rgba(*col); c.fill_preserve()
    c.set_source_rgba(0.45, 0.52, 0.6, 0.35); c.set_line_width(4); c.stroke()
    if turn > 0:  # голова начинает поворачиваться — край бледного лица
        c.save()
        c.arc(0, -540, 72, 0, 2 * math.pi); c.clip()
        c.arc(72 - 40 * turn, -540, 60, 0, 2 * math.pi)
        c.set_source_rgba(*hexc("#bdb7ad", 0.9)); c.fill()
        c.arc(56 - 40 * turn, -550, 6, 0, 2 * math.pi)
        c.set_source_rgba(0.05, 0.05, 0.05, turn); c.fill()
        c.restore()
    c.restore()


def window_frame(c, x, y, w, h, t, moon=0.35, seed=3):
    c.rectangle(x, y, w, h)
    c.set_source(lin(c, 0, y, 0, y + h, [(0, hexc("#2e3946")), (1, hexc("#1d242c"))]))
    c.fill()
    # лес за окном
    rng = np.random.default_rng(seed)
    c.save(); c.rectangle(x, y, w, h); c.clip()
    for i in range(9):
        xx = x + rng.uniform(0, w)
        c.rectangle(xx, y + h * 0.15, rng.uniform(8, 22), h)
        c.set_source_rgba(0.06, 0.08, 0.1, 0.9); c.fill()
    c.restore()
    c.rectangle(x, y, w, h)
    c.set_source_rgba(*hexc("#151618")); c.set_line_width(14); c.stroke()
    c.move_to(x + w / 2, y); c.line_to(x + w / 2, y + h); c.stroke()
    c.move_to(x, y + h * 0.42); c.line_to(x + w, y + h * 0.42); c.stroke()


def sc_inside(c, t, lt, T, ev):
    bg(c, "#141518", "#0b0b0c")
    # стена
    c.rectangle(0, 0, W, 1250); c.set_source_rgba(*hexc("#25282c")); c.fill()
    for xx in (60, 420, 780):
        glow(c, xx + 125, 640, 330, (0.45, 0.55, 0.68), 0.28)
        window_frame(c, xx, 430, 250, 380, t, seed=int(xx))
    ground(c, 1250, "#1d1f22", "#0b0b0c")
    # лунный свет из окон на полу
    for xx in (60, 420, 780):
        c.move_to(xx, 1250); c.line_to(xx + 250, 1250); c.line_to(xx + 330, 1700); c.line_to(xx - 60, 1700)
        c.close_path()
        c.set_source(lin(c, 0, 1250, 0, 1700, [(0, (0.5, 0.6, 0.72, 0.16)), (1, (0.5, 0.6, 0.72, 0))])); c.fill()
    # люди спиной у окон
    people = [(185, 1330, 0.95, False), (545, 1350, 1.02, True), (905, 1335, 0.98, False),
              (360, 1420, 1.1, False), (740, 1430, 1.08, False)]
    for i, (px, py, s, scarf) in enumerate(people):
        turn = 0.0
        if i == 4 and lt > T - 1.4:
            turn = ease((lt - (T - 1.4)) / 1.2)
        person_back(c, px, py, s, scarf, turn)
    # луч фонарика ведёт по спинам
    fl = flashlight_level(t, ev)
    sx = 150 + ease(lt / T) * 780
    c.set_operator(cairo.OPERATOR_ADD)
    glow(c, sx, 1080 + smooth_noise(t, 1, 2) * 30, 360, (0.55, 0.53, 0.48), 0.75 * fl)
    c.set_operator(cairo.OPERATOR_OVER)


def sc_granny(c, t, lt, T, ev):
    bg(c, "#121315", "#0a0a0b")
    zoom = 1 + 0.12 * ease(lt / T)
    shake = smooth_noise(t, 9, 14) * 4
    c.save()
    c.translate(W / 2 + shake, 900); c.scale(zoom, zoom)
    # платок
    c.move_to(-330, 520)
    c.curve_to(-380, 0, -300, -470, 0, -480)
    c.curve_to(300, -470, 380, 0, 330, 520)
    c.close_path()
    fill_stroke(c, hexc("#2b2826"))
    c.set_source_rgba(0, 0, 0, 0.25); c.set_line_width(3)
    for i in range(-3, 4):
        c.move_to(i * 60, -460); c.curve_to(i * 75, -100, i * 90, 200, i * 95, 500); c.stroke()
    # лицо
    c.move_to(-210, -150)
    c.curve_to(-220, -380, 220, -380, 210, -150)
    c.curve_to(205, 150, 110, 320, 0, 330)
    c.curve_to(-110, 320, -205, 150, -210, -150)
    c.close_path()
    c.set_source(lin(c, 0, 330, 0, -380, [(0, hexc("#d9d2c6")), (1, hexc("#8f8a82"))]))  # свет снизу
    c.fill_preserve(); c.set_source_rgba(*OUT); c.set_line_width(5); c.stroke()
    # морщины
    c.set_source_rgba(0.2, 0.18, 0.17, 0.55); c.set_line_width(3)
    for yy in (-250, -225, -200):
        c.move_to(-110, yy); c.curve_to(-40, yy - 12, 40, yy - 12, 110, yy); c.stroke()
    for sx in (-1, 1):
        c.move_to(sx * 70, 90); c.curve_to(sx * 95, 150, sx * 90, 200, sx * 70, 240); c.stroke()
    # глазницы и зрачки
    wide = 1 + 0.25 * ease((t - ev["whisper"]) / 0.4) if t > ev["whisper"] else 1
    for sx in (-1, 1):
        c.save(); c.translate(sx * 85, -110); c.scale(1, 0.62 * wide)
        c.arc(0, 0, 62, 0, 2 * math.pi)
        c.set_source(rad(c, 0, 0, 10, 62, [(0, hexc("#0b0b0c")), (1, hexc("#3a3632"))])); c.fill()
        c.restore()
        c.arc(sx * 85 + smooth_noise(t, sx, 3) * 3, -112, 9, 0, 2 * math.pi)
        c.set_source_rgba(0.92, 0.9, 0.85, 1); c.fill()
    # нос
    c.set_source_rgba(0.25, 0.22, 0.2, 0.6); c.set_line_width(4)
    c.move_to(-10, -70); c.curve_to(-30, 20, -35, 40, 0, 50); c.stroke()
    # рот — шепчет
    mo = 0.35 + 0.65 * abs(math.sin(t * 11)) if ev["whisper"] < t < ev["whisper_end"] else 0.2
    c.save(); c.translate(0, 175); c.scale(1, 0.25 + 0.5 * mo)
    c.arc(0, 0, 55, 0, 2 * math.pi); c.set_source_rgba(*hexc("#0d0b0b")); c.fill()
    c.restore()
    c.restore()
    # костлявая рука сжимает рукав героя (рукав идёт по диагонали снизу)
    c.save(); c.translate(540, 1830 - 40 * (1 - ease(lt / 0.4))); c.rotate(-0.12)
    rrect(c, -700, -95, 1400, 190, 90); fill_stroke(c, hexc("#4c5157"))
    c.set_source_rgba(*hexc("#363a3f")); c.set_line_width(4)
    for xx in range(-600, 700, 120):
        c.move_to(xx, -92); c.curve_to(xx + 20, -30, xx + 20, 30, xx, 92); c.stroke()
    # тыльная сторона ладони
    c.move_to(-150, -150); c.curve_to(-60, -210, 90, -200, 150, -130)
    c.line_to(130, -60); c.curve_to(40, -95, -60, -95, -140, -60); c.close_path()
    fill_stroke(c, hexc("#bdb5a9"), lw=4)
    # пальцы обхватывают рукав
    for i, fx in enumerate((-115, -45, 25, 92)):
        ln = 150 + (12 if i in (1, 2) else 0)
        c.save(); c.translate(fx, -75); c.rotate(0.08 * (i - 1.5))
        rrect(c, -21, -10, 42, ln, 20); fill_stroke(c, hexc("#c3bbaf"), lw=4)
        c.set_source_rgba(0.35, 0.3, 0.27, 0.8); c.set_line_width(3)
        for yy in (ln * 0.35, ln * 0.68):
            c.move_to(-12, yy); c.curve_to(-4, yy + 5, 4, yy + 5, 12, yy); c.stroke()
        c.arc(0, ln - 26, 11, 0, math.pi)  # ноготь
        c.set_source_rgba(0.55, 0.5, 0.45, 1); c.fill()
        c.restore()
    # сухожилия
    c.set_source_rgba(0.45, 0.4, 0.36, 0.7); c.set_line_width(3)
    for fx in (-110, -40, 30, 95):
        c.move_to(fx * 0.4, -180); c.line_to(fx, -90); c.stroke()
    c.restore()


def sc_follower(c, t, lt, T, ev):
    bg(c, "#191b1e", "#0e0f11")
    forest_layers(c, t, drift=0, fog_a=0.9, seed=11)
    ground(c, 1260)
    z = 1 + 0.1 * ease(lt / T)
    a = 0.2 + 0.55 * ease(lt / (T * 0.8))
    c.save(); c.translate(640, 1270); c.scale(z, z)
    col = (0.62, 0.62, 0.6, a)
    # длинная худая фигура между деревьями
    c.set_source_rgba(*col)
    c.move_to(-26, 0); c.line_to(-20, -520); c.curve_to(-50, -560, -40, -640, 0, -650)
    c.curve_to(40, -640, 50, -560, 20, -520); c.line_to(26, 0); c.close_path(); c.fill()
    c.arc(0, -700, 38, 0, 2 * math.pi); c.fill()
    c.set_line_width(12); c.set_line_cap(cairo.LINE_CAP_ROUND)
    for sx in (-1, 1):
        c.move_to(sx * 34, -620); c.curve_to(sx * 70, -420, sx * 66, -250, sx * 80, -60); c.stroke()
    c.set_source_rgba(0.03, 0.03, 0.03, a * 1.2)
    for sx in (-1, 1):
        c.arc(sx * 13, -705, 6, 0, 2 * math.pi); c.fill()
    c.restore()
    # ближние стволы перекрывают
    c.set_source_rgba(*hexc("#0f1012"))
    c.rectangle(470, -50, 84, 1400); c.fill()
    c.rectangle(830, -50, 120, 1400); c.fill()
    c.rectangle(80, -50, 64, 1400); c.fill()


def sc_window(c, t, lt, T, ev):
    """Вид из окна на дом напротив. Щелчок — фонарь гаснет; затем в доме напротив зажигается свет."""
    on = t >= ev["light_on"]
    k_on = ease((t - ev["light_on"]) / 0.12) if on else 0
    z = 1 + 0.18 * ease(lt / T)
    c.save()
    c.translate(W / 2, H * 0.48); c.scale(z, z); c.translate(-W / 2, -H * 0.48)
    bg(c, "#0d0e10", "#08090a")
    # улица за окном
    c.save(); c.rectangle(170, 380, 740, 1000); c.clip()
    c.set_source(lin(c, 0, 380, 0, 1380, [(0, hexc("#1f252c")), (1, hexc("#161a1f"))])); c.paint()
    glow(c, 760, 520, 160, (0.7, 0.72, 0.75), 0.25)
    c.arc(760, 520, 40, 0, 2 * math.pi); c.set_source_rgba(*hexc("#77787a")); c.fill()
    ground(c, 1080, "#2a2e33", "#1a1d21")
    house(c, 560, 1120, 420, 300, 150, wall=hexc("#2b2f34"), lit=k_on, figure=k_on, win2=True)
    c.restore()
    # рама окна изнутри
    c.rectangle(170, 380, 740, 1000)
    c.set_source_rgba(*hexc("#121314")); c.set_line_width(40); c.stroke()
    c.set_line_width(22)
    c.move_to(540, 380); c.line_to(540, 1380); c.stroke()
    c.move_to(170, 800); c.line_to(910, 800); c.stroke()
    # подоконник
    c.rectangle(110, 1380, 860, 60); fill_stroke(c, hexc("#18191b"))
    c.restore()
    # фонарь в комнате до щелчка
    fl = flashlight_level(t, ev)
    if fl > 0:
        c.set_operator(cairo.OPERATOR_ADD)
        glow(c, 520, 1500, 520, (0.45, 0.43, 0.4), 0.5 * fl)
        c.set_operator(cairo.OPERATOR_OVER)
    # после щелчка всё темнеет; до включения света — почти полная тьма
    if t >= ev["click"] and not on:
        k = ease((t - ev["click"]) / 0.15)
        c.set_source_rgba(0, 0, 0, 0.55 * k); c.paint()


SCENES = {"forest": sc_forest, "village": sc_village, "door": sc_door, "inside": sc_inside,
          "granny": sc_granny, "follower": sc_follower, "window": sc_window}


# ------------------------------------------------------------------ captions
def caption(c, words, k, y=1430, size=66):
    """words: [(text, accent)] — строка субтитров в стиле референса."""
    c.select_font_face("Inter ExtraBold", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(size)
    space = c.text_extents(" ").x_advance or size * 0.28
    # перенос на 2 строки, если широко
    lines, cur, cw = [], [], 0
    for wd, acc in words:
        ww = c.text_extents(wd).x_advance
        if cur and cw + space + ww > W - 160:
            lines.append(cur); cur, cw = [], 0
        cur.append((wd, acc, ww)); cw += (space if len(cur) > 1 else 0) + ww
    if cur:
        lines.append(cur)
    sc = 0.9 + 0.1 * ease(k / 0.6)
    a = ease(k)
    c.save()
    c.translate(W / 2, y); c.scale(sc, sc)
    for li, line in enumerate(lines):
        tw = sum(w for _, _, w in line) + space * (len(line) - 1)
        x = -tw / 2
        yy = (li - (len(lines) - 1) / 2) * size * 1.12 + size * 0.35
        for wd, acc, ww in line:
            c.move_to(x + 3, yy + 5)
            c.text_path(wd)
            c.set_source_rgba(0, 0, 0, 0.75 * a); c.set_line_width(10)
            c.set_line_join(cairo.LINE_JOIN_ROUND); c.stroke_preserve(); c.fill()
            c.move_to(x, yy)
            c.text_path(wd)
            col = RED if acc else hexc("#f1f1ee")
            c.set_source_rgba(col[0], col[1], col[2], a); c.fill()
            x += ww + space
    c.restore()


def chapter(c, label):
    c.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(27)
    c.set_source_rgba(0.62, 0.62, 0.6, 0.85)
    x = 72
    for ch in "— " + label:
        c.move_to(x, 268); c.show_text(ch)
        x += c.text_extents(ch).x_advance + 3.5


# ------------------------------------------------------------------ audio
def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def compress_gaps(audio, words, rules):
    """Укорачивает слишком длинные паузы между словами; возвращает новый звук и сдвинутые тайминги."""
    out, pos, shift = [], 0, 0.0
    new_words = [dict(w) for w in words]
    for i in range(len(words) - 1):
        end = words[i]["t"] + words[i]["d"]
        nxt = words[i + 1]["t"]
        keep = rules.get(i + 1, rules["default"])
        gap = nxt - end
        if gap > keep + 0.05:
            cut = gap - keep
            a = int((end + (gap - cut) / 2) * SR)
            b = a + int(cut * SR)
            out.append(audio[pos:a]); pos = b
            shift += cut
        new_words[i + 1]["t"] = words[i + 1]["t"] - shift
    out.append(audio[pos:])
    # мягкая склейка: короткий кроссфейд не нужен — режем в тишине
    return np.concatenate(out), new_words


def bp(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, btype="high", fs=SR, output="sos"), x)


def env_curve(n, pts):
    """Огибающая по точкам [(t, value)]."""
    ts = np.arange(n) / SR
    xs, ys = zip(*pts)
    return np.interp(ts, xs, ys)


def place(buf, snd, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(snd))
    buf[i:j] += snd[: j - i] * gain


def sound_design(total, ev, voice_env):
    n = int(total * SR)
    ts = np.arange(n) / SR
    L = np.zeros(n); R = np.zeros(n)

    def brown(m, seed):
        w = np.random.default_rng(seed).standard_normal(m)
        b = hp(np.cumsum(w), 15)
        return b / (np.abs(b).max() + 1e-9)

    # ветер (лес, деревня) — стерео
    for ch, seed in ((L, 1), (R, 2)):
        wind = bp(brown(n, seed), 180, 1400)
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * ts + seed) * np.sin(2 * np.pi * 0.047 * ts)
        e = env_curve(n, [(0, 0), (0.8, 1), (ev["inside"] - 0.3, 1), (ev["inside"], 0.25),
                          (ev["follower"], 0.25), (ev["follower"] + 0.3, 0.9), (ev["window"], 0.9),
                          (ev["window"] + 0.2, 0.35), (ev["click"], 0.35), (ev["click"] + 0.02, 0.05),
                          (total, 0.05)])
        ch += wind * lfo * e * 0.22

    # низкий гул
    f0 = 43.65
    drone = (np.sin(2 * np.pi * f0 * ts) + 0.8 * np.sin(2 * np.pi * f0 * 1.004 * ts) +
             0.45 * np.sin(2 * np.pi * f0 * 2.01 * ts) + 0.2 * np.sin(2 * np.pi * f0 * 2.38 * ts))
    de = env_curve(n, [(0, 0), (2, 0.6), (ev["inside"], 0.8), (ev["granny"], 1.0), (ev["click"], 1.0),
                       (ev["click"] + 0.03, 0.25), (ev["light_on"], 0.35), (ev["light_on"] + 0.05, 0.0),
                       (total, 0.0)])
    d = drone * de * 0.16 * (0.85 + 0.15 * np.sin(2 * np.pi * 0.2 * ts))
    L += d; R += d

    # диссонансный кластер, растущее напряжение
    cl = sum(np.sin(2 * np.pi * f * ts + p) for f, p in ((622, 0), (659, 1), (698, 2), (740, 3)))
    cl *= 0.7 + 0.3 * np.sin(2 * np.pi * 5.5 * ts)
    ce = env_curve(n, [(0, 0), (ev["granny"], 0), (ev["follower"], 0.35), (ev["follower"] + 2.5, 0.6),
                       (ev["window"], 0.3), (ev["click"], 0.3), (ev["click"] + 0.03, 0),
                       (ev["window2"], 0), (ev["light_on"] - 0.05, 0.9), (ev["light_on"], 0), (total, 0)])
    cl = bp(cl, 400, 2500) * ce * 0.035
    L += cl; R += np.roll(cl, 400)

    # шаги по снегу в лесу
    step = 0.62
    t0 = 0.5
    while t0 < ev["village"] - 0.3:
        m = int(0.12 * SR)
        g = np.random.default_rng(int(t0 * 100)).standard_normal(m)
        g = bp(g, 900, 5000) * np.exp(-np.linspace(0, 6, m)) * (np.random.default_rng(int(t0)).random(m) > 0.55)
        place(L, g, t0, 0.18); place(R, g, t0 + 0.004, 0.14)
        t0 += step

    # щелчки мигающего фонаря
    for i in range(7):
        tt = ev["flicker"] - 0.15 + i * 0.19 + (i % 3) * 0.03
        m = int(0.012 * SR)
        g = bp(np.random.default_rng(i).standard_normal(m), 2000, 8000) * np.exp(-np.linspace(0, 8, m))
        place(L, g, tt, 0.25); place(R, g, tt, 0.25)

    # скрип двери
    m = int(1.8 * SR)
    tt = np.arange(m) / SR
    rate = 35 + 55 * np.sin(np.pi * tt / 1.8) + 10 * np.sin(2 * np.pi * 3 * tt)
    phase = np.cumsum(rate) / SR
    imp = (np.diff(np.floor(phase), prepend=0) > 0).astype(float) * (0.6 + 0.4 * np.random.default_rng(3).random(m))
    k = np.arange(int(0.02 * SR)) / SR
    res = (np.sin(2 * np.pi * 850 * k) + 0.6 * np.sin(2 * np.pi * 1430 * k) + 0.3 * np.sin(2 * np.pi * 2900 * k)) * np.exp(-k * 160)
    creak = signal.fftconvolve(imp, res)[:m] * np.sin(np.pi * tt / 1.8) ** 0.5
    creak /= np.abs(creak).max() + 1e-9
    place(L, creak, ev["door"] + 0.25, 0.22); place(R, creak, ev["door"] + 0.27, 0.16)

    # часы-ходики в доме
    tt = ev["inside"] + 0.4
    while tt < ev["follower"]:
        m = int(0.03 * SR)
        g = bp(np.random.default_rng(int(tt * 10)).standard_normal(m), 2500, 6000) * np.exp(-np.linspace(0, 10, m))
        place(L, g, tt, 0.09); place(R, g, tt, 0.07)
        tt += 1.0

    # сердцебиение, ускоряется
    tt, bpm = ev["granny"], 62.0
    while tt < ev["click"] - 0.2:
        for off, amp in ((0, 1.0), (0.26, 0.7)):
            m = int(0.25 * SR)
            k = np.arange(m) / SR
            f = 58 - 20 * k / 0.25
            thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 18)
            place(L, thump, tt + off, 0.55 * amp); place(R, thump, tt + off, 0.55 * amp)
        tt += 60 / bpm
        bpm = min(118, bpm + 4.5)

    # щелчок фонаря
    m = int(0.05 * SR)
    k = np.arange(m) / SR
    click = (np.random.default_rng(9).standard_normal(m) * np.exp(-k * 600) * 0.7 +
             np.sin(2 * np.pi * 2100 * k) * np.exp(-k * 250) * 0.5)
    place(L, click, ev["click"], 0.6); place(R, click, ev["click"], 0.6)

    # финальный удар + визг
    m = int(3.2 * SR)
    k = np.arange(m) / SR
    boom = np.sin(2 * np.pi * np.cumsum(80 * np.exp(-k * 1.3) + 28) / SR) * np.exp(-k * 1.4)
    boom += lp(np.random.default_rng(5).standard_normal(m), 300) * np.exp(-k * 6) * 2
    scr = sum(signal.sawtooth(2 * np.pi * f * k + 0.8 * np.sin(2 * np.pi * (6 + i) * k)) for i, f in
              enumerate((1180, 1252, 1331, 1409, 1497)))
    scr = bp(scr, 900, 5000) * np.exp(-k * 1.1) * np.minimum(1, k / 0.015)
    hit = boom / (np.abs(boom).max() + 1e-9) * 0.9 + scr / (np.abs(scr).max() + 1e-9) * 0.35
    place(L, hit, ev["light_on"], 0.95); place(R, np.roll(hit, 300), ev["light_on"], 0.95)

    # приглушение фона под голосом
    duck = 1 - 0.45 * voice_env
    return np.stack([L * duck, R * duck], axis=1)


def reverb(x, seconds=0.9, wet=0.14):
    m = int(seconds * SR)
    k = np.arange(m) / SR
    ir = np.random.default_rng(4).standard_normal(m) * np.exp(-k * 6.5)
    ir = lp(ir, 3500)
    y = signal.fftconvolve(x, ir)[: len(x)]
    y /= np.abs(y).max() + 1e-9
    return x + y * wet * np.abs(x).max()


# ------------------------------------------------------------------ main
def align_tokens(text, words):
    toks = [tk for tk in text.split() if re.search(r"\w", tk)]
    norm = lambda s: re.sub(r"[^\wё]", "", s.lower())
    out, j = [], 0
    for w in words:
        while j < len(toks) and norm(toks[j]) != norm(w["w"]):
            j += 1
        out.append(toks[j] if j < len(toks) else w["w"])
        j += 1
    return out


def chunk_words(n, display, scene_starts, max_words=3):
    chunks, cur = [], []
    for i in range(n):
        if cur and (i in scene_starts or len(cur) >= max_words):
            chunks.append(cur); cur = []
        cur.append(i)
        if re.search(r"[.!?…,»]$", display[i]):
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story")
    ap.add_argument("-o", "--out", default="horror.mp4")
    ap.add_argument("--preview", type=float, default=None, help="отрендерить только кадр в момент t (png)")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()

    story = json.loads(Path(a.story).read_text(encoding="utf-8"))
    work = Path(a.work or tempfile.mkdtemp(prefix="horror_"))
    work.mkdir(parents=True, exist_ok=True)
    txt = work / "story.txt"
    txt.write_text(story["text"], encoding="utf-8")

    # 1. озвучка
    raw_mp3, words_json = work / "voice_raw.mp3", work / "words.json"
    if not words_json.exists():
        env = dict(os.environ, VOICE=story.get("voice", "ru-RU-DmitryNeural"),
                   RATE=story.get("rate", "-5%"), PITCH=story.get("pitch", "-6Hz"))
        subprocess.run([sys.executable, str(HERE / "tts_words.py"), str(txt), str(raw_mp3), str(words_json)],
                       check=True, env=env)
    words = json.loads(words_json.read_text())
    voice = load_audio(raw_mp3)
    rules = {"default": story.get("max_gap", 0.75)}
    rules.update({int(k): v for k, v in story.get("gaps", {}).items()})
    voice, words = compress_gaps(voice, words, rules)
    display = align_tokens(story["text"], words)

    vo_off = story.get("voice_offset", 0.7)
    for w in words:
        w["t"] += vo_off
    voice_end = vo_off + len(voice) / SR
    total = voice_end + story.get("tail", 2.6)

    # события
    scenes = [(s["at"], s["scene"], s["label"]) for s in story["scenes"]]
    st = lambda wi: words[wi]["t"] - 0.12
    ev = {name: (0.0 if wi == 0 else st(wi)) for wi, name, _ in scenes}
    seen = {}
    for wi, name, _ in scenes:  # повторяющиеся сцены: window, window2
        if name in seen:
            ev[name + "2"] = st(wi)
        seen[name] = 1
    for key, wi in story["events"].items():
        ev[key] = words[wi]["t"]
    ev.setdefault("window2", ev["window"])
    timeline = [(0.0 if wi == 0 else st(wi), name, label) for wi, name, label in scenes]

    accents = {x.lower() for x in story.get("accents", [])}
    if story.get("chunks"):
        chunks, i0 = [], 0
        for nwd in story["chunks"]:
            chunks.append(list(range(i0, i0 + nwd))); i0 += nwd
    else:
        chunks = chunk_words(len(words), display, {wi for wi, _, _ in scenes})
    caps = []
    for ch in chunks:
        t0 = words[ch[0]]["t"] - 0.05
        caps.append((t0, [(display[i].upper().lstrip("—"),
                           re.sub(r"[^\wё-]", "", display[i].lower()) in accents) for i in ch]))

    caps_last_t = caps[-1][0]

    def caption_at(t):
        cur = None
        for i, (t0, ws) in enumerate(caps):
            if t >= t0:
                cur = (t0, ws)
        if cur and (t - cur[0] < 2.6 or cur[0] == caps_last_t):
            return cur
        return None

    # 2. звук
    voice_track = np.zeros(int(total * SR))
    v = reverb(voice * 1.0)
    place(voice_track, v / (np.abs(v).max() + 1e-9) * 0.8, vo_off)
    venv = np.convolve(np.abs(voice_track), np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
    venv = np.clip(venv / (venv.max() + 1e-9) * 2.2, 0, 1)
    amb = sound_design(total, ev, venv)
    mix = amb + voice_track[:, None]
    mix = np.tanh(mix * 1.1) / 1.1
    wav = work / "mix.wav"
    wavfile.write(wav, SR, (mix / (np.abs(mix).max() + 1e-9) * 0.95).astype(np.float32))

    # 3. видео
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.47) / (H * 0.6)) ** 2)
    vign = np.clip(1.08 - r ** 2.2 * 0.9, 0.18, 1.0)[..., None].astype(np.float32)
    grains = [(gaussian_filter(RNG.standard_normal((H // 2, W // 2)), 0.6) * 9).astype(np.float32)
              for _ in range(6)]
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    c = cairo.Context(surf)

    def render(t, fi):
        idx = max(i for i, (t0, _, _) in enumerate(timeline) if t >= t0 or i == 0)
        t0, name, label = timeline[idx]
        t1 = timeline[idx + 1][0] if idx + 1 < len(timeline) else total
        c.save()
        c.set_source_rgb(0, 0, 0); c.paint()
        # лёгкий наезд камеры + дрожь
        z = 1 + 0.04 * ((t - t0) / max(t1 - t0, 0.1))
        c.translate(W / 2 + smooth_noise(t, 1, 0.7) * 5, H / 2 + smooth_noise(t, 2, 0.6) * 5)
        c.scale(z, z); c.translate(-W / 2, -H / 2)
        SCENES[name](c, t, t - t0, t1 - t0, ev)
        c.restore()
        chapter(c, label)
        cap = caption_at(t)
        if cap:
            big = story.get("big_last") and cap[0] == caps_last_t
            caption(c, cap[1], (t - cap[0]) / 0.18, y=story.get("caption_y", 1530) - (60 if big else 0),
                    size=130 if big else 66)
        surf.flush()
        buf = np.ndarray((H, W, 4), np.uint8, surf.get_data())[..., :3].astype(np.float32)
        # затемнение на склейках
        fade = min(1.0, (t - t0) / 0.22) if idx > 0 else min(1.0, t / 0.6)
        fade = min(fade, 1.0 if t1 - t > 0.08 else (t1 - t) / 0.08 + 0.0) if idx + 1 < len(timeline) else fade
        if t > total - 0.35:
            fade = min(fade, (total - t) / 0.35)
        flick = 1 - 0.035 * (0.5 + 0.5 * math.sin(t * 31 + math.sin(t * 7)))
        g = grains[fi % len(grains)]
        g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
        buf = (buf * vign * flick * max(0.0, fade) + g)
        # лёгкая холодная тонировка
        buf[..., 0] *= 1.03  # B (порядок BGR)
        return np.clip(buf, 0, 255).astype(np.uint8)

    if a.preview is not None:
        img = render(a.preview, 0)
        out = Path(a.out)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                        "-i", "-", "-frames:v", "1", str(out)], input=img.tobytes(), check=True)
        print(out)
        return

    nfr = int(total * FPS)
    silent = work / "video.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
    for fi in range(nfr):
        p.stdin.write(render(fi / FPS, fi).tobytes())
        if fi % 150 == 0:
            print(f"кадр {fi}/{nfr}", file=sys.stderr)
    p.stdin.close(); p.wait()

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav),
                    "-af", "loudnorm=I=-14:TP=-1.0:LRA=9", "-ar", "48000",
                    "-vf", "hqdn3d=0:0:3:3", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", a.out],
                   check=True)
    print(f"готово: {a.out} ({total:.1f} c)")


if __name__ == "__main__":
    main()
