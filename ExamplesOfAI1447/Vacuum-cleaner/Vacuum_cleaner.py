class VacuumCleaner:
    def __init__(self):
        self.location = 'A'# الموقع الابتدائي
        self.environment = {'A': 'Dirty', 'B': 'Dirty'}# البيئة مع حالة كل موقع
    
    def perceive(self):# إدراك الحالة الحالية
        return (self.location, self.environment[self.location])# إرجاع الموقع والحالة
    
    def act(self):# اتخاذ إجراء بناءً على الإدراك
        location, status = self.perceive()# الحصول على الإدراك الحالي
        if status == 'Dirty':# إذا كان المكان متسخًا
            self.environment[location] = 'Clean' # تنظيف المكان
            return 'Suck' # إرجاع الإجراء وهو الشفط
        elif location == 'A':# مالم يكن متسخا و كان في الموقع A
            self.location = 'B' # الانتقال إلى الموقع B
            return 'Right' # إرجاع الإجراء وهو الانتقال إلى اليمين
        else: # مالم يكن متسخا و كان في الموقع B
            self.location = 'A' # الانتقال إلى الموقع A
            return 'Left' # إرجاع الإجراء وهو الانتقال إلى اليسار

# اختبار الوكيل
vacuum = VacuumCleaner() # إنشاء وكيل التنظيف
print("Initial Status:", vacuum.environment) # طباعة الحالة الابتدائية

for i in range(6): # تنفيذ 6 خطوات
    action = vacuum.act() # اتخاذ إجراء
    print(f"Step {i+1}: Action = {action}, State = {vacuum.environment}") # طباعة الإجراء والحالة بعد كل خطوة
