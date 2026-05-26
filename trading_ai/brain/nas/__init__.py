"""
Neural Architecture Search (NAS) — AI ค้นหา architecture ที่ดีที่สุดเอง

ทำไม AI ถึงต้องทำ NAS?
  มนุษย์ออกแบบ neural network architecture ด้วยการลองผิดลองถูก
  แต่ AI ทำได้เร็วกว่า เป็นระบบกว่า และไม่เหนื่อย

กลไก:
  1. Champion  — brain หลักที่เทรดจริง
  2. Challengers — shadow brains ที่เรียนรู้คู่ขนาน ไม่เทรดจริง
  3. Evaluation — เปรียบความแม่นยำทุก 100 trades
  4. Evolution  — challenger ที่ดีกว่า → แนะนำให้ upgrade
  5. Mutation   — สร้าง challenger ใหม่จากการ evolve config

เหนือกว่ามนุษย์:
  → มนุษย์ออกแบบ architecture ครั้งเดียว แล้วใช้ไปตลอด
  → AI นี้ออกแบบ architecture ตัวเองใหม่ทุกๆ 100 trades
"""
from trading_ai.brain.nas.nas_engine import NASEngine

__all__ = ["NASEngine"]
