import pandas as pd
import numpy as np


class EV_profile(object):
    def __init__(self):
        self.order = None
        self.old_status = None
        self.status = None
        self.chg_status = None
        self.home = None
        self.trip = None
        self.c_lo = None
        self.n_lo = None
        self.p_dt = None
        self.dis = None
        self.travel_decision = None
        self.chg_lo = None
        self.start_t = None
        self.arrive_t = None
        self.total_trip = None
        self.cbus_lo = None
        self.nbus_lo = None
        self.work = None
        self.type = None
        self.batt = None
        self.econ = None
        self.soc = None
        self.auten = None
        self.old_chg_status = None
        self.number = None
        self.d = None
        self.t = None
        self.power = {}


class EV_data(object):
    def __init__(self):
        self.grid_position = None
        self.location = None
        self.park_dt = None
        self.behave = None
        self.perform = None
        self.EV_profile = {}
        self.number = None

    def download_EV(self):
        self.perform = pd.read_excel('EVdata.xlsx', sheet_name='EVtype')
        self.behave = pd.read_excel('EVdata.xlsx', sheet_name='behavior')
        self.park_dt = pd.read_excel('EVdata.xlsx', sheet_name='park')
        self.location = pd.read_excel('EVdata.xlsx', sheet_name='location')
        self.grid_position = pd.read_excel('EVdata.xlsx', sheet_name='grid')

    def cal_geomatic_grid(self):
        x = self.grid_position.location.shape[0]
        self.grid_dis = np.zeros([x, x])

        # position[i,j]
        for i in range(x):
            for j in range(x):
                self.grid_dis[i, j] = self.cal_distance(i, j)

    def cal_distance(self, i, j):
        i_x = self.grid_position.at[i, 'x']
        i_y = self.grid_position.at[i, 'y']
        j_x = self.grid_position.at[j, 'x']
        j_y = self.grid_position.at[j, 'y']
        dis = (abs(i_x - j_x) + abs(i_y - j_y)) * 5
        if dis == 0:
            dis = 2.5

        return dis

    def create_ev(self, number):
        for i in range(number):
            self.number = number
            ev = EV_profile()
            ev.order = i + 1
            self.EV_profile[i] = ev

    def setting_ev_profile(self):
        for i in range(self.number):
            # random type
            cal = self.perform.type.dropna()
            cal = cal.to_numpy()
            cal2 = self.perform.ProbBattery.dropna()
            cal2 = cal2.to_numpy()
            self.EV_profile[i].type = np.random.choice(cal, 1, p=cal2)[0]
            self.EV_profile[i].econ = self.perform.loc[self.perform.type == self.EV_profile[i].type, 'EnergyCon'].iat[0]
            self.EV_profile[i].batt = self.perform.loc[self.perform.type == self.EV_profile[i].type, 'BatteryCap'].iat[
                0]
            self.EV_profile[i].power[1] = self.perform.loc[self.perform.type == self.EV_profile[i].type, 'power74'].iat[
                0]
            self.EV_profile[i].power[2] = self.perform.loc[self.perform.type == self.EV_profile[i].type, 'power22'].iat[
                0]
            self.EV_profile[i].power[3] = self.perform.loc[self.perform.type == self.EV_profile[i].type, 'power50'].iat[
                0]
            self.EV_profile[i].soc = 1
            self.EV_profile[i].chg_status = 0

            # random home
            cal = self.location.home.dropna()
            cal = cal.to_numpy()
            cal2 = self.location.probhome.dropna()
            cal2 = cal2.to_numpy()
            self.EV_profile[i].home = np.random.choice(cal, 1, p=cal2)[0]

            # set currentlocation
            self.EV_profile[i].cbus_lo = self.EV_profile[i].home
            self.EV_profile[i].c_lo = 1
            self.EV_profile[i].trip = 1

            # set initial start time
            cal = np.arange(48)
            cal2 = self.behave.probtdepart.dropna()
            cal2 = cal2.to_numpy()
            self.EV_profile[i].start_t = np.random.choice(cal, 1, p=cal2)[0]
            self.EV_profile[i].status = 0
            self.EV_profile[i].d = 0

            # set total trip
            cal = np.arange(1, 5)
            cal2 = self.behave.probtrip.dropna()
            cal2 = cal2.to_numpy()
            self.EV_profile[i].total_trip = np.random.choice(cal, 1, p=cal2)[0]

            # set next bus location
            cal = self.location.work.dropna()
            cal = cal.to_numpy()
            cal2 = self.location.probwork.dropna()
            cal2 = cal2.to_numpy()
            self.EV_profile[i].nbus_lo = np.random.choice(cal, 1, p=cal2)[0]
            self.EV_profile[i].n_lo = 2
            self.EV_profile[i].work = self.EV_profile[i].nbus_lo

            # initial charging location
            cal = np.array([1, 2, 3])
            cal2 = np.array([0.80, 0.15, 0.05])
            self.EV_profile[i].chg_lo = np.random.choice(cal, 1, p=cal2)[0]


class charging_matrix(object):
    def __init__(self, bus, power, time):
        array = [bus, power, time]
        array2 = [power, 2]
        self.event = np.zeros(array)
        self.chg_dt = np.zeros(array2)
        self.park_dt = np.zeros(array2)


class log(object):
    def __init__(self, t, ev):
        array = [t, ev]
        self.soc = np.zeros(array)
        self.bus_lo = np.zeros(array)
        self.power = np.zeros(array)
        self.chg_lo = np.zeros(array)
        self.chg_status = np.zeros(array)
        self.auten = np.zeros(array)
        self.p_dt = np.zeros(array)
        self.c_lo = np.zeros(array)
        self.trip = np.zeros(array)
        self.chg_power = np.zeros(array)


class EV_profile_test(object):
    def __init__(self):
        self.c_lo = None
        self.stop_t = None
        self.c_dt = None
        self.pc_dt = None


class sim_data(object):

    def __init__(self):
        self.EV_profile = []
        self.number = None

    def create_ev(self, number):
        for i in range(number):
            self.number = number
            ev = EV_profile_test()
            ev.order = i + 1
            ev.c_dt = 0
            ev.pc_dt = 0
            self.EV_profile.append(ev)
