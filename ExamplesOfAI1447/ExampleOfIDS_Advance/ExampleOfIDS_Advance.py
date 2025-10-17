# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from collections import defaultdict, deque
import time
import random

# === إعدادات دعم النص العربي ===
try:
    # لمستخدمي Windows
    os.system('chcp 65001')
except:
    pass

# إعادة تكوين الإخراج ليدعم UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# محاولة استيراد مكتبات تحسين النص العربي
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    
    def format_arabic(text):
        """تنسيق النص العربي"""
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
except ImportError:
    # إذا لم تكن المكتبات مثبتة، استخدم النص كما هو
    def format_arabic(text):
        return text

class IntelligentSecurityAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        # معاملات التعلم
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # تخزين التاريخ
        self.perception_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        
        # تعلم الأنماط
        self.ip_behavior_patterns = defaultdict(lambda: defaultdict(int))
        self.protocol_patterns = defaultdict(lambda: defaultdict(int))
        self.threat_patterns = defaultdict(int)
        
        # جدول Q-learning
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # الإجراءات المتاحة - استخدام نصوص إنجليزية لتجنب مشاكل الترميز
        self.actions = ["MONITOR", "ALERT_ADMIN", "BLOCK_IP"]
        self.threat_states = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        # العتبات الديناميكية
        self.dynamic_thresholds = {
            'packet_count': 1000,
            'icmp_threshold': 100,
            'connection_rate': 50,
            'suspicious_pattern': 5
        }
        
        # الإحصاءات
        self.stats = {
            'total_threats_detected': 0,
            'false_positives': 0,
            'learning_improvements': 0,
            'total_decisions': 0
        }

    def perceive(self, network_traffic):
        """استشعار البيئة وجمع البيانات مع السياق التاريخي"""
        basic_percept = {
            'source_ip': network_traffic['src_ip'],
            'destination_ip': network_traffic['dst_ip'],
            'packet_count': network_traffic['packet_count'],
            'protocol': network_traffic['protocol'],
            'timestamp': time.time()
        }
        
        # إضافة السياق التاريخي
        historical_context = self._get_historical_context(network_traffic)
        enriched_percept = {**basic_percept, **historical_context}
        
        self.perception_history.append(enriched_percept)
        return enriched_percept

    def _get_historical_context(self, current_traffic):
        """الحصول على السياق التاريخي للحركة الحالية"""
        source_ip = current_traffic['src_ip']
        
        # تحليل السلوك التاريخي لهذا IP
        recent_activities = [
            p for p in self.perception_history 
            if p['source_ip'] == source_ip
        ]
        
        # تجنب القسمة على صفر
        avg_packet = np.mean([p['packet_count'] for p in recent_activities]) if recent_activities else 1.0
        
        historical_context = {
            'recent_activity_count': len(recent_activities),
            'avg_packet_count': avg_packet,
            'behavior_consistency': self._calculate_behavior_consistency(source_ip),
            'threat_frequency': self._calculate_threat_frequency(source_ip),
            'time_since_last_activity': self._get_time_since_last_activity(source_ip)
        }
        
        return historical_context

    def _calculate_behavior_consistency(self, ip):
        """حساب اتساق السلوك بناءً على التاريخ"""
        ip_activities = [p for p in self.perception_history if p['source_ip'] == ip]
        if len(ip_activities) < 2:
            return 1.0
            
        packet_counts = [p['packet_count'] for p in ip_activities]
        
        # تجنب القسمة على صفر
        if np.mean(packet_counts) == 0:
            return 1.0
            
        consistency = 1.0 / (1.0 + np.std(packet_counts) / np.mean(packet_counts))
        return float(np.clip(consistency, 0, 1))

    def _calculate_threat_frequency(self, ip):
        """حساب تكرار التهديدات من IP معين"""
        threat_count = 0
        for i, action in enumerate(self.action_history):
            if i < len(self.perception_history):
                percept = self.perception_history[i]
                if percept['source_ip'] == ip and action in ["ALERT_ADMIN", "BLOCK_IP"]:
                    threat_count += 1
        return threat_count

    def _get_time_since_last_activity(self, ip):
        """الحصول على الوقت منذ آخر نشاط"""
        ip_activities = [p for p in self.perception_history if p['source_ip'] == ip]
        if not ip_activities:
            return float('inf')
        
        last_activity = max(ip_activities, key=lambda x: x['timestamp'])
        return time.time() - last_activity['timestamp']

    def analyze_threat_with_learning(self, percepts):
        """تحليل التهديد مع التعلم من التاريخ"""
        # القواعد الأساسية
        base_threat = self._base_threat_analysis(percepts)
        
        # التعلم المعزز للتهديد
        learned_threat = self._reinforcement_learning_analysis(percepts)
        
        # تحليل الأنماط
        pattern_threat = self._pattern_based_analysis(percepts)
        
        # الجمع بين جميع التحليلات
        combined_threat = self._combine_threat_assessments(base_threat, learned_threat, pattern_threat)
        
        # التأكد من أن القيمة بين 0-3 وتحويل إلى integer
        return int(np.clip(round(combined_threat), 0, 3))

    def _base_threat_analysis(self, percepts):
        """التحليل الأساسي للتهديد"""
        threat_score = 0
        
        if percepts['packet_count'] > self.dynamic_thresholds['packet_count']:
            threat_score += 2
        
        if percepts['protocol'] == 'ICMP' and percepts['packet_count'] > self.dynamic_thresholds['icmp_threshold']:
            threat_score += 1
        
        if percepts['recent_activity_count'] > self.dynamic_thresholds['connection_rate']:
            threat_score += 1
            
        if percepts['behavior_consistency'] < 0.3:
            threat_score += 1
            
        if percepts['threat_frequency'] > self.dynamic_thresholds['suspicious_pattern']:
            threat_score += 2
            
        return threat_score

    def _reinforcement_learning_analysis(self, percepts):
        """تحليل التهديد باستخدام التعلم المعزز"""
        state = self._get_state_representation(percepts)
        
        if np.random.random() < self.exploration_rate:
            return random.randint(0, 3)
        else:
            if state in self.q_table and self.q_table[state]:
                return int(max(self.q_table[state].items(), key=lambda x: x[1])[0])
            else:
                return 0

    def _pattern_based_analysis(self, percepts):
        """تحليل التهديد بناءً على أنماط التعلم"""
        source_ip = percepts['source_ip']
        protocol = percepts['protocol']
        
        # تجنب القسمة على صفر
        total_activities = max(len(self.perception_history), 1)
        protocol_score = self.protocol_patterns[source_ip][protocol] / total_activities
        
        avg_packet = max(percepts['avg_packet_count'], 1)  # تجنب القسمة على صفر
        packet_anomaly = abs(percepts['packet_count'] - avg_packet) / avg_packet
        
        time_anomaly = 1.0 if percepts['time_since_last_activity'] < 60 else 0.5
        
        threat_score = protocol_score + packet_anomaly + time_anomaly
        return min(threat_score, 3)

    def _get_state_representation(self, percepts):
        """تمثيل الحالة للتعلم المعزز"""
        packet_group = int(percepts['packet_count'] / 100)
        activity_group = int(percepts['recent_activity_count'] / 10)
        return f"{percepts['source_ip']}_{percepts['protocol']}_{packet_group}_{activity_group}"

    def _combine_threat_assessments(self, base, learned, pattern):
        """الجمع بين تقييمات التهديد المختلفة"""
        weights = self._calculate_dynamic_weights()
        combined = (base * weights['base'] + 
                   learned * weights['learned'] + 
                   pattern * weights['pattern'])
        
        return combined

    def _calculate_dynamic_weights(self):
        """حساب الأوزان الديناميكية بناءً على دقة التعلم"""
        total_decisions = len(self.action_history)
        if total_decisions < 10:
            return {'base': 0.6, 'learned': 0.2, 'pattern': 0.2}
        
        success_rate = self._calculate_learning_success_rate()
        learned_weight = 0.2 + (success_rate * 0.6)
        base_weight = 0.6 - (success_rate * 0.4)
        
        return {'base': base_weight, 'learned': learned_weight, 'pattern': 0.2}

    def _calculate_learning_success_rate(self):
        """حساب معدل نجاح التعلم من التاريخ"""
        if len(self.reward_history) < 5:
            return 0.5
            
        positive_rewards = sum(1 for r in self.reward_history if r > 0)
        return positive_rewards / len(self.reward_history)

    def act_with_learning(self, threat_level, percepts):
        """اتخاذ إجراء مع التعلم من النتائج"""
        state = self._get_state_representation(percepts)
        
        # التأكد من أن threat_level هو integer
        threat_level = int(threat_level)
        
        # اختيار الإجراء بناءً على مستوى التهديد والتعلم
        action_index = self._select_action_based_on_threat(threat_level, state)
        action = self.actions[action_index]
        
        # تحديث Q-table بناءً على الإجراء المختار
        self._update_q_table(state, action_index, threat_level)
        
        self.action_history.append(action)
        self.stats['total_decisions'] += 1
        return action

    def _select_action_based_on_threat(self, threat_level, state):
        """اختيار الإجراء المناسب بناءً على مستوى التهديد"""
        threat_level = int(threat_level)  # التأكد من أنه integer
        
        if threat_level == 0:  # LOW
            return 0  # MONITOR
        elif threat_level == 1:  # MEDIUM
            return 1  # ALERT_ADMIN
        elif threat_level == 2:  # HIGH
            if state in self.q_table and self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                return min(int(best_action), 2)
            else:
                return 1  # ALERT_ADMIN افتراضي
        else:  # CRITICAL (3)
            return 2  # BLOCK_IP

    def _calculate_reward(self, action_index, threat_level):
        """حساب المكافأة بناءً على مناسبة الإجراء لمستوى التهديد"""
        action_severity = int(action_index)
        threat_level = int(threat_level)
        
        optimal_match = threat_level == action_severity
        over_reaction = action_severity > threat_level
        under_reaction = action_severity < threat_level
        
        if optimal_match:
            return 1.0
        elif over_reaction:
            return -0.5
        elif under_reaction:
            return -1.0
        else:
            return 0.0

    def _update_q_table(self, state, action_index, threat_level):
        """تحديث جدول Q-learning"""
        reward = self._calculate_reward(action_index, threat_level)
        self.reward_history.append(reward)
        
        current_q = self.q_table[state][action_index]
        max_future_q = max(self.q_table[state].values()) if self.q_table[state] else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action_index] = new_q

    def learn_from_feedback(self, percepts, action_taken, was_correct):
        """التعلم من التغذية الراجعة الخارجية"""
        threat_level = self.analyze_threat_with_learning(percepts)
        state = self._get_state_representation(percepts)
        action_index = self.actions.index(action_taken)
        
        if was_correct:
            reward = 2.0
            self.stats['learning_improvements'] += 1
        else:
            reward = -2.0
            self.stats['false_positives'] += 1
        
        # تحديث مباشر لـ Q-table بناءً على التغذية الراجعة
        current_q = self.q_table[state][action_index]
        self.q_table[state][action_index] = current_q + self.learning_rate * reward
        
        self._update_dynamic_thresholds(was_correct)

    def _update_dynamic_thresholds(self, was_correct):
        """تحديث العتبات الديناميكية بناءً على الأداء"""
        if was_correct:
            self.dynamic_thresholds['packet_count'] = max(500, self.dynamic_thresholds['packet_count'] - 10)
        else:
            self.dynamic_thresholds['packet_count'] = min(2000, self.dynamic_thresholds['packet_count'] + 50)

    def get_learning_stats(self):
        """الحصول على إحصائيات التعلم"""
        return {
            'total_decisions': self.stats['total_decisions'],
            'learning_success_rate': self._calculate_learning_success_rate(),
            'recent_positive_rewards': sum(1 for r in list(self.reward_history)[-10:] if r > 0),
            'dynamic_thresholds': self.dynamic_thresholds.copy(),
            'performance_stats': self.stats.copy(),
            'q_table_size': len(self.q_table),
            'memory_usage': len(self.perception_history)
        }

    def _calculate_confidence(self):
        """حساب ثقة الوكيل في قراراته"""
        if len(self.reward_history) < 5:
            return 0.5
            
        recent_rewards = list(self.reward_history)[-10:]
        if not recent_rewards:
            return 0.5
            
        positive_ratio = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
        return positive_ratio

    def run_intelligent_agent(self, network_traffic, feedback=None):
        """تشغيل الوكيل الذكي الكامل"""
        # 1. الإدراك مع السياق التاريخي
        percepts = self.perceive(network_traffic)
        
        # 2. التحليل مع التعلم - التأكد من أن threat_level هو integer
        threat_level = self.analyze_threat_with_learning(percepts)
        threat_level = int(threat_level)  # تحويل إلى integer
        
        # 3. اتخاذ الإجراء مع التعلم
        action = self.act_with_learning(threat_level, percepts)
        
        # 4. التعلم من التغذية الراجعة إذا وجدت
        if feedback is not None:
            self.learn_from_feedback(percepts, action, feedback)
            self.stats['total_threats_detected'] += 1
        
        return {
            'action': action,
            'threat_level': self.threat_states[threat_level],  # الآن threat_level integer
            'threat_score': threat_level,
            'confidence': self._calculate_confidence(),
            'learning_stats': self.get_learning_stats(),
            'historical_context': {
                'recent_activities': percepts['recent_activity_count'],
                'behavior_consistency': round(percepts['behavior_consistency'], 2),
                'threat_frequency': percepts['threat_frequency'],
                'avg_packet_count': round(percepts['avg_packet_count'], 2)
            }
        }

# محاكي حركة المرور
class NetworkTrafficSimulator:
    def __init__(self):
        self.normal_ips = ['192.168.1.10', '192.168.1.11', '192.168.1.12']
        self.suspicious_ips = ['10.0.0.100', '10.0.0.101']
        self.protocols = ['TCP', 'UDP', 'ICMP', 'HTTP']
        
    def generate_normal_traffic(self):
        return {
            'src_ip': random.choice(self.normal_ips),
            'dst_ip': f'10.0.0.{random.randint(1, 50)}',
            'packet_count': random.randint(1, 100),
            'protocol': random.choice(self.protocols)
        }
    
    def generate_suspicious_traffic(self):
        traffic_type = random.choice(['ddos', 'scan', 'flood'])
        
        if traffic_type == 'ddos':
            return {
                'src_ip': random.choice(self.suspicious_ips),
                'dst_ip': '10.0.0.1',
                'packet_count': random.randint(1000, 5000),
                'protocol': 'TCP'
            }
        elif traffic_type == 'scan':
            return {
                'src_ip': random.choice(self.suspicious_ips),
                'dst_ip': f'10.0.0.{random.randint(1, 255)}',
                'packet_count': random.randint(150, 300),
                'protocol': 'ICMP'
            }
        else:  # flood
            return {
                'src_ip': random.choice(self.suspicious_ips),
                'dst_ip': f'10.0.0.{random.randint(1, 10)}',
                'packet_count': random.randint(800, 2000),
                'protocol': random.choice(['UDP', 'TCP'])
            }

# اختبار النظام الكامل
if __name__ == "__main__":
    # استخدام النص العربي مع التنسيق
    print(format_arabic("=== بدء تشغيل النظام الأمني الذكي ==="))
    print()
    
    # إنشاء الوكيل والمحاكي
    smart_agent = IntelligentSecurityAgent()
    simulator = NetworkTrafficSimulator()
    
    # عدد دورات المحاكاة
    simulation_cycles = 20
    
    print(format_arabic("محاكاة حركة مرور الشبكة..."))
    print("-" * 60)
    
    for cycle in range(simulation_cycles):
        print(format_arabic(f"\n--- الدورة {cycle + 1} ---"))
        
        # توليد حركة مرور (80% عادية، 20% مشبوهة)
        if random.random() < 0.8:
            traffic = simulator.generate_normal_traffic()
            actual_threat = False
        else:
            traffic = simulator.generate_suspicious_traffic()
            actual_threat = True
        
        print(f"حركة المرور: {traffic}")
        
        # محاكاة التغذية الراجعة
        feedback = actual_threat if random.random() < 0.9 else not actual_threat
        
        # تشغيل الوكيل
        result = smart_agent.run_intelligent_agent(traffic, feedback)
        
        # عرض النتائج باستخدام النص العربي
        print(format_arabic(f"→ الإجراء: {result['action']}"))
        print(format_arabic(f"→ مستوى التهديد: {result['threat_level']} (درجة: {result['threat_score']})"))
        print(format_arabic(f"→ الثقة: {result['confidence']:.2f}"))
        print(format_arabic(f"→ السياق: {result['historical_context']}"))
        
        # عرض التقدم في التعلم كل 5 دورات
        if (cycle + 1) % 5 == 0:
            stats = result['learning_stats']
            print(format_arabic(f"\n📊 تقرير التعلم بعد {cycle + 1} دورة:"))
            print(format_arabic(f"   - القرارات الكلية: {stats['total_decisions']}"))
            print(format_arabic(f"   - معدل النجاح: {stats['learning_success_rate']:.2f}"))
            print(format_arabic(f"   - المكافآت الإيجابية الأخيرة: {stats['recent_positive_rewards']}/10"))
            print(format_arabic(f"   - عتبة الحزم: {stats['dynamic_thresholds']['packet_count']}"))
            print(format_arabic(f"   - حجم ذاكرة Q: {stats['q_table_size']} حالة"))
    
    print("\n" + "=" * 60)
    print(format_arabic("=== التقرير النهائي ==="))
    
    final_stats = smart_agent.get_learning_stats()
    
    print(format_arabic("\nإحصائيات الأداء:"))
    for stat_key, stat_value in final_stats['performance_stats'].items():
        print(format_arabic(f"  - {stat_key}: {stat_value}"))
    
    print(format_arabic("\nالعتبات الديناميكية النهائية:"))
    for threshold_key, threshold_value in final_stats['dynamic_thresholds'].items():
        print(format_arabic(f"  - {threshold_key}: {threshold_value}"))
    
    print(format_arabic("\nخلاصة أداء الوكيل:"))
    total_decisions = final_stats['performance_stats']['total_decisions']
    learning_improvements = final_stats['performance_stats']['learning_improvements']
    false_positives = final_stats['performance_stats']['false_positives']
    
    if total_decisions > 0:
        improvement_rate = (learning_improvements / total_decisions) * 100
        false_positive_rate = (false_positives / total_decisions) * 100
        
        print(format_arabic(f"معدل التحسن: {improvement_rate:.1f}%"))
        print(format_arabic(f"معدل الإنذارات الكاذبة: {false_positive_rate:.1f}%"))
        print(format_arabic(f"معدل نجاح التعلم: {final_stats['learning_success_rate']:.1%}"))
    
    print(format_arabic("\n🎯 النظام جاهز للاستخدام في بيئة حقيقية!"))