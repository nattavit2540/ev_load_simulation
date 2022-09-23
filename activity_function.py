import numpy as np
import math
import pandas as pd


# check status
def status_calculation(ev, t, n_day):
    for i in range(ev.number):
        # update status
        ev.EV_profile[i].old_status = ev.EV_profile[i].status
        ev.EV_profile[i].t = t
        if ev.EV_profile[i].status == 0 and ev.EV_profile[i].start_t == t:
            ev.EV_profile[i].status = 1
        elif ev.EV_profile[i].status == 1 and ev.EV_profile[i].arrive_t == t:
            ev.EV_profile[i].status = 0

        # check parking/driving
        # parking
        if ev.EV_profile[i].status == 0:
            # arriving
            if ev.EV_profile[i].arrive_t == t:
                # check charging need
                ev.EV_profile[i].c_lo = ev.EV_profile[i].n_lo
                ev.EV_profile[i].cbus_lo = ev.EV_profile[i].nbus_lo

                # parking duartion
                if (ev.EV_profile[i].trip == ev.EV_profile[i].total_trip):
                    if ev.EV_profile[i].d < n_day:
                        ev.EV_profile[i].d = n_day
                        ev.EV_profile[i].trip = 1
                        ev.EV_profile[i].n_lo = 2
                        ev.EV_profile[i].nbus_lo = ev.EV_profile[i].work

                        # set total trip
                        cal = np.arange(1, 5)
                        cal2 = ev.behave.probtrip.dropna()
                        cal2 = cal2.to_numpy()
                        ev.EV_profile[i].total_trip = np.random.choice(cal, 1, p=cal2)[0]

                        # set initial start time
                        cal = np.arange(48)
                        cal2 = ev.behave.probtdepart.dropna()
                        cal2 = cal2.to_numpy()
                        ev.EV_profile[i].start_t = (np.random.choice(cal, 1, p=cal2)[0]) + (ev.EV_profile[i].d * 48)

                    else:
                        ev.EV_profile[i].d = n_day + 1
                        ev.EV_profile[i].trip = 1
                        ev.EV_profile[i].n_lo = 2
                        ev.EV_profile[i].nbus_lo = ev.EV_profile[i].work

                        # set total trip
                        cal = np.arange(1, 5)
                        cal2 = ev.behave.probtrip.dropna()
                        cal2 = cal2.to_numpy()
                        ev.EV_profile[i].total_trip = np.random.choice(cal, 1, p=cal2)[0]

                        # set initial start time
                        cal = np.arange(48)
                        cal2 = ev.behave.probtdepart.dropna()
                        cal2 = cal2.to_numpy()
                        ev.EV_profile[i].start_t = (np.random.choice(cal, 1, p=cal2)[0]) + (ev.EV_profile[i].d * 48)

                    if ev.EV_profile[i].start_t <= t:
                        ev.EV_profile[i].start_t += 48

                elif (ev.EV_profile[i].trip == ev.EV_profile[i].total_trip - 1):
                    if (ev.EV_profile[i].c_lo == 3):
                        ev.EV_profile[i].trip = ev.EV_profile[i].trip + 1
                        ev.EV_profile[i].n_lo = 1
                        ev.EV_profile[i].nbus_lo = ev.EV_profile[i].home

                        # cal parking duration for public charging station
                        cal = (1 - ev.EV_profile[i].soc) * ev.EV_profile[i].batt / ev.EV_profile[i].power[3] * 2
                        ev.EV_profile[i].start_t = math.ceil(ev.EV_profile[i].arrive_t + cal)


                    else:
                        # chcek charging need
                        if ev.EV_profile[i].chg_lo == 3:
                            ev.EV_profile[i].n_lo = 3

                            # random charging location
                            cal = ev.location.pub.dropna()
                            cal = cal.to_numpy()
                            cal2 = ev.location.probpub.dropna()
                            cal2 = cal2.to_numpy()
                            ev.EV_profile[i].nbus_lo = np.random.choice(cal, 1, p=cal2)[0]

                            # cal parking duration
                            cal = ev.EV_profile[i].trip
                            cal2 = ev.park_dt[cal].dropna()
                            cal2 = cal2.to_numpy()
                            cal = np.arange(48)
                            ev.EV_profile[i].p_dt = np.random.choice(cal, 1, p=cal2)[0]
                            ev.EV_profile[i].start_t = ev.EV_profile[i].arrive_t + ev.EV_profile[i].p_dt

                        else:
                            # cal parking duration
                            cal = ev.EV_profile[i].trip
                            cal2 = ev.park_dt[cal].dropna()
                            cal2 = cal2.to_numpy()
                            cal = np.arange(48)
                            ev.EV_profile[i].p_dt = np.random.choice(cal, 1, p=cal2)[0]
                            ev.EV_profile[i].start_t = ev.EV_profile[i].arrive_t + ev.EV_profile[i].p_dt
                            ev.EV_profile[i].trip += 1
                            ev.EV_profile[i].n_lo = 1
                            ev.EV_profile[i].nbus_lo = ev.EV_profile[i].home
                else:

                    # cal parking duration
                    cal = ev.EV_profile[i].trip
                    cal2 = ev.park_dt[cal].dropna()
                    cal2 = cal2.to_numpy()
                    cal = np.arange(48)
                    ev.EV_profile[i].p_dt = np.random.choice(cal, 1, p=cal2)[0]
                    ev.EV_profile[i].start_t = ev.EV_profile[i].arrive_t + ev.EV_profile[i].p_dt
                    ev.EV_profile[i].trip += 1
                    ev.EV_profile[i].n_lo = 4

                    # random other location
                    cal = ev.location.other.dropna()
                    cal = cal.to_numpy()
                    cal2 = ev.location.probother.dropna()
                    cal2 = cal2.to_numpy()
                    ev.EV_profile[i].nbus_lo = np.random.choice(cal, 1, p=cal2)[0]

            if (ev.EV_profile[i].c_lo == ev.EV_profile[i].chg_lo) and (ev.EV_profile[i].soc) < 1:
                ev.EV_profile[i].chg_status = 1
            else:
                ev.EV_profile[i].chg_status = 0

        # driving
        elif ev.EV_profile[i].status == 1:

            if ev.EV_profile[i].start_t == t:
                # cal SoC usage
                # random distance
                ev.EV_profile[i].chg_status = 0

                # calculate from cbus_lo and nbus_lo
                ev.EV_profile[i].dis = ev.grid_dis[int(ev.EV_profile[i].cbus_lo - 1), int(ev.EV_profile[i].nbus_lo - 1)]
                if ev.EV_profile[i].dis == 0:
                    print(i, t, ev.EV_profile[i].nbus_lo)

                # cal Arriving time
                ev.EV_profile[i].arrive_t = math.ceil(
                    ev.EV_profile[i].start_t + (ev.EV_profile[i].dis / ev.behave.at[t % 48, 'velocity'] * 2))

                # calculate SoC
                # energy con
                ev.EV_profile[i].soc = ev.EV_profile[i].soc - (
                            (ev.EV_profile[i].econ * ev.EV_profile[i].dis) / ev.EV_profile[i].batt)
                if ev.EV_profile[i].trip == 1 and ev.EV_profile[i].c_lo == 1:
                    cal = np.array([1, 2, 3])
                    cal2 = np.array([0.80, 0.15, 0.05])
                    ev.EV_profile[i].chg_lo = np.random.choice(cal, 1, p=cal2)[0]


def autenti_ev(ev, control, t, Flagsheet=None):
    sort = None
    Flag = Flagsheet

    if control == 1:
        for i in range(ev.number):
            ev.EV_profile[i].auten = 1

    elif control == 2:
        for i in range(ev.number):
            ev.EV_profile[i].auten = 0

    elif control == 3:
        sort = {}
        sort['order'] = []
        sort['soc'] = []

        for idx, veh in enumerate(ev.EV_profile):
            if veh.chg_status == 1:
                sort['order'].append(veh.order)
                sort['soc'].append(veh.order)

        sort = pd.DataFrame.from_dict(sort)
        sort = sort.sort_values('soc')

        for idx, row in sort.iterrows():
            cal = ev.EV_profile[row['order'] - 1].cbus_lo
            di1 = ev.grid_position.loc[ev.grid_position.location == cal, 'bus']

            # di2
            if ev.EV_profile[row['order'] - 1].chg_lo == 1:
                di2 = 0
            elif ev.EV_profile[row['order'] - 1].chg_lo == 2:
                di2 = 1
            elif ev.EV_profile[row['order'] - 1].chg_lo == 3:
                di2 = 2

            # di3
            di3 = t % 48
            di1 = int(di1)
            di2 = int(di2)
            di3 = int(di3)
            if ev.EV_profile[row['order'] - 1].soc < 1 and ev.EV_profile[row['order'] - 1].chg_status == 1:
                if Flag[di1, di3] > 0:
                    ev.EV_profile[row['order'] - 1].auten = 1
                    Flag[di1, di3] = Flag[di1, di3] - 1
    return sort


def datatrack(ev, matrix, t):
    # in this function : tracking each EV charging Event
    di1 = 0
    di2 = 0
    di3 = 0
    # di1 = bus di2=power di3 = time
    for i in range(ev.number):
        # positioning
        if ev.EV_profile[i].chg_status == 1 and ev.EV_profile[i].old_status == 1:
            # di1
            cal = ev.EV_profile[i].cbus_lo
            di1 = ev.grid_position.loc[ev.grid_position.location == cal, 'bus']

            # di2
            # di2
            if ev.EV_profile[i].chg_lo == 1:
                di2 = 0
            elif ev.EV_profile[i].chg_lo == 2:
                di2 = 1
            elif ev.EV_profile[i].chg_lo == 3:
                di2 = 2

            # di3
            di3 = t % 48

            di1 = int(di1)
            di2 = int(di2)
            di3 = int(di3)

            matrix.event[di1, di2, di3] = matrix.event[di1, di2, di3] + 1
            matrix.chg_dt[di2, 0] = ((1 - ev.EV_profile[i].soc) * 2 * ev.EV_profile[i].batt / ev.EV_profile[i].power[
                di2 + 1]) + matrix.chg_dt[di2, 0]
            matrix.chg_dt[di2, 1] += 1

            matrix.park_dt[di2, 0] = (ev.EV_profile[i].start_t - ev.EV_profile[i].arrive_t) + matrix.park_dt[di2, 0]
            matrix.park_dt[di2, 1] += 1


def charging_ev(ev, p, t, log):
    for i in range(ev.number):
        if (ev.EV_profile[i].chg_status == 1) and (ev.EV_profile[i].auten == 1):
            # Charging Power
            if ev.EV_profile[i].chg_lo == 1:
                ev_power = ev.EV_profile[i].power[1]
            elif ev.EV_profile[i].chg_lo == 2:
                ev_power = ev.EV_profile[i].power[2]
            elif ev.EV_profile[i].chg_lo == 3:
                ev_power = ev.EV_profile[i].power[3]

            # Calculate suitable charging duration

            r_dt = (1 - ev.EV_profile[i].soc) * ev.EV_profile[i].batt / ev_power * 2

            if r_dt < 1:
                ev.EV_profile[i].soc = 1
                ch_dt = r_dt
            else:
                ev.EV_profile[i].soc = ev.EV_profile[i].soc + (ev_power / 2 / ev.EV_profile[i].batt)
                ch_dt = 1

            # Power demand calculation
            load = ev_power * (ch_dt)
            cal = ev.EV_profile[i].cbus_lo
            cal2 = ev.grid_position.loc[ev.grid_position.location == cal, 'bus']
            p[t, cal2] = p[t, cal2] + load
        #     log.chg_power[t,i] = load
        #
        # log.soc[t,i] = ev.EV_profile[i].soc
        # log.bus_lo[t,i] = ev.EV_profile[i].cbus_lo
        # log.chg_lo[t,i] = ev.EV_profile[i].chg_lo
        # log.chg_status[t,i] = ev.EV_profile[i].chg_status
        # log.auten[t,i] = ev.EV_profile[i].auten
        # log.p_dt[t,i] = ev.EV_profile[i].p_dt
        # log.c_lo[t,i] = ev.EV_profile[i].c_lo
        # log.trip[t,i] = ev.EV_profile[i].trip
