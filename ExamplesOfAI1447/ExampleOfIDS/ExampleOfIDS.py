class SecurityAgent:
    def __init__(self):
        self.suspicious_activities = []
        self.blocked_ips = []
    
    def perceive(self, network_traffic):
        # محاكاة تحليل حركة المرور
        return {
            'source_ip': network_traffic['src_ip'],
            'destination_ip': network_traffic['dst_ip'],
            'packet_count': network_traffic['packet_count'],
            'protocol': network_traffic['protocol']
        }
    
    def analyze_threat(self, percepts):
        # تحليل التهديدات بناءً على القواعد
        if percepts['packet_count'] > 1000:  # هجوم حجب الخدمة
            return 'HIGH'
        elif percepts['protocol'] == 'ICMP' and percepts['packet_count'] > 100:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def act(self, threat_level, source_ip):
        if threat_level == 'HIGH':
            self.blocked_ips.append(source_ip)
            return f"BLOCK_IP {source_ip}"
        elif threat_level == 'MEDIUM':
            return f"ALERT_ADMIN {source_ip}"
        else:
            return "MONITOR"

# اختبار الوكيل الأمني
security_bot = SecurityAgent()
sample_traffic = {'src_ip': '192.168.1.100', 'dst_ip': '10.0.0.1', 
                  'packet_count': 1500, 'protocol': 'TCP'}

percepts = security_bot.perceive(sample_traffic)
threat = security_bot.analyze_threat(percepts)
action = security_bot.act(threat, sample_traffic['src_ip'])

print(f"Threat detected: {threat}")
print(f"Action taken: {action}")
