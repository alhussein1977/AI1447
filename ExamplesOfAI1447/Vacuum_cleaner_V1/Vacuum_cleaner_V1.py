#استيراد كل ما يتعلق بمكتبات tkinter والتي تضم مكتبة Tk()
from tkinter import *
from agents import * 

#سقوم بتعريف الموقعين A,B واعطاهم قيم اولية 
loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world
#نقوم بتعريف كلاس Gui و
#Environment لن يتعرف عليها وهي اصلن كلاس موجود في ملف agents
#لذا سنحتاج الى استيراد كل ما في ملف agents وذلك عبر السطر from agents import *
class Gui(Environment):
    """بيئة واجهة المستخدم الرسومية هذه لها موقعان ، A و B. كل يمكن أن تكون قذرة
        أو نظيفة. يدرك الوكيل موقعه وموقعه  الحالة."""

    def __init__(self, root, height=300, width=380):
        super().__init__()
        #يعطي البيئة حالة للموقعين
        self.status = {loc_A: 'Clean',
                       loc_B: 'Clean'}
        self.root = root
        self.height = height
        self.width = width
        self.canvas = None
        self.buttons = []
        self.create_canvas()
        self.create_buttons()
    def thing_classes(self):
        """The list of things which can be used in the environment."""
        return [Wall, Dirt, ReflexVacuumAgent, RandomVacuumAgent,
                TableDrivenVacuumAgent, ModelBasedVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        """Change the location status (Dirty/Clean); track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                if agent.location == loc_A:
                    self.buttons[0].config(bg='white', activebackground='light grey')
                else:
                    self.buttons[1].config(bg='white', activebackground='light grey')
                agent.performance += 10
            self.status[agent.location] = 'Clean'

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])

    def create_canvas(self):
        """ينشئ عنصر Canvas في واجهة المستخدم الرسومية."""
        self.canvas = Canvas(
            self.root,
            width=self.width,
            height=self.height,
            background='powder blue')
        self.canvas.pack(side='bottom')
    def create_buttons(self):
        """ينشئ الأزرار المطلوبة في واجهة المستخدم الرسومية."""
        button_left = Button(self.root, height=4, width=12, padx=2, pady=2, bg='white')
        button_left.config(command=lambda btn=button_left: self.dirt_switch(btn))
        self.buttons.append(button_left)
        button_left_window = self.canvas.create_window(130, 200, anchor=N, window=button_left)
        button_right = Button(self.root, height=4, width=12, padx=2, pady=2, bg='white')
        button_right.config(command=lambda btn=button_right: self.dirt_switch(btn))
        self.buttons.append(button_right)
        button_right_window = self.canvas.create_window(250, 200, anchor=N, window=button_right)
    def dirt_switch(self, button):
        """يمنح المستخدم خيار وضع الأوساخ في أي بلاطة."""
        bg_color = button['bg']
        if bg_color == 'saddle brown':
            button.config(bg='white', activebackground='light grey')
        elif bg_color == 'white':
            button.config(bg='saddle brown', activebackground='light goldenrod')
    
    def read_env(self):
        """Reads the current state of the GUI."""
        for i, btn in enumerate(self.buttons):
            if i == 0:
                if btn['bg'] == 'white':
                    self.status[loc_A] = 'Clean'
                else:
                    self.status[loc_A] = 'Dirty'
            else:
                if btn['bg'] == 'white':
                    self.status[loc_B] = 'Clean'
                else:
                    self.status[loc_B] = 'Dirty'

    def update_env(self, agent):
        """Updates the GUI according to the agent's action."""
        self.read_env()
        # print(self.status)
        before_step = agent.location
        self.step()
        # print(self.status)
        # print(agent.location)
        move_agent(self, agent, before_step)
def create_agent(env, agent):
    """ينشئ الوكيل في واجهة المستخدم الرسومية ويتم الاحتفاظ به مستقلا عن البيئة."""
    env.add_thing(agent)
    # print(agent.location)
    if agent.location == (0, 0):
        env.agent_rect = env.canvas.create_rectangle(80, 100, 175, 180, fill='lime green')
        env.text = env.canvas.create_text(128, 140, font="Helvetica 10 bold italic", text="Agent")
    else:
        env.agent_rect = env.canvas.create_rectangle(200, 100, 295, 180, fill='lime green')
        env.text = env.canvas.create_text(248, 140, font="Helvetica 10 bold italic", text="Agent")
def move_agent(env, agent, before_step):
    """Moves the agent in the GUI when 'next' button is pressed."""
    if agent.location == before_step:
        pass
    else:
        if agent.location == (1, 0):
            env.canvas.move(env.text, 120, 0)
            env.canvas.move(env.agent_rect, 120, 0)
        elif agent.location == (0, 0):
            env.canvas.move(env.text, -120, 0)
            env.canvas.move(env.agent_rect, -120, 0)
# TODO: Add more agents to the environment.
# TODO: Expand the environment to XYEnvironment.
if __name__ == "__main__":
    #تعريف متغير باسم root من نوع مكتبة Tk 
    #مكتبة Tk هي مكتبة تعمل على اظهار نوافذ window
    #لن تعمل السطر التالي الا باستيراد مكتبة tkinter لذا سنقوم باضافة المكتبة في اول سطر
    root = Tk()
    #اعطى النافذة عنوان
    root.title("Vacuum Environment")
    #اعطي النافذة حجم
    root.geometry("420x380")
    #اعطي النافذة القيم 0 و 0 لاعادة حجمهم
    root.resizable(0, 0)
    #قام بتعريف fram من نوع Frame وهذا الاطار سيكون من هو الاب له وهو root وخلفية لون اسود black
    frame = Frame(root, bg='black')
    # reset_button = Button(frame, text='Reset', height=2, width=6, padx=2, pady=2, command=None)
    # reset_button.pack(side='left')
    #قام بتعريف زر التالي وهو من نوع Button وحدد من هو الاب و النص الذي يظر وطوله وعرضه ...الخ
    next_button = Button(frame, text='Next', height=2, width=6, padx=2, pady=2)
    #حدد ان زر التالي يقع بالجانب الايسر
    next_button.pack(side='left')
    #حدد ان الاطار frame يقع بالاسفل
    frame.pack(side='bottom')
    #هنا عرف متغير اسمه env وبانه يساوي Gui(root)
    #Gui هي كلاس سنقوم بانشاءه الان هذا الكلاس من نوع Environment وهي بيئة العمل , وهي root في مثالنا هذا
    #نقوم بكتابة الكلاس Gui بنفس هذا الملف
    env = Gui(root)
    #هنا بنحتاج الى دالة ReflexVacuumAgent وهي موجودة في ملف agent
    agent = ReflexVacuumAgent()
    create_agent(env, agent)
    next_button.config(command=lambda: env.update_env(agent))
    root.mainloop()


