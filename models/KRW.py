class KRW:

    def __init__(self, K=1, dt=0.01, T=10, N=None, w=None, theta=None, eps=1):
        '''
        K: float
            Constante de acople, default = 1
        dt: float
            Delta t para integrar, default = 0.01
        T: float
            Tiempo total, default = 10
        N: int, opcional
            Número de osciladores
            Se puede deducir de la lontitud de w o theta
            Requerido si no se recibe w ni theta
        w: 1D ndarray, opcional
            Frecuencia natural de los osciladores
            Si no se recibe un valor, se inician todos en 0. Previamente se generaba al azar con distribución normal estándar
            Requerido si no se recibe N ni theta
        theta: 1D ndarray, opcional
            Fase inicial de los osciladores
            Si no se recibe un valor, se genera al azar uniformemente en [0,2*pi]
            Requerido si no se recibe N ni w
        eps: float, opcional
            Tiempo promedio de salto de los paseos al azar, default = 1
        '''
        if N is None and w is None and theta is None:
            raise ValueError("Error: falta un valor de N, w o theta")
        if w is not None and theta is not None:
            assert len(theta) == len(w), f"La dimension de w: {len(w)} no coincide con la de theta: {len(theta)}"
        if w is not None and N is not None:
            assert N == len(w), f"La dimension de w: {len(w)} no coincide con N"
        if N is not None and theta is not None:
            assert len(theta) == N, f"La dimension de theta: {len(theta)} no coincide con N"

        self.dt = dt
        self.T = T
        self.K = K
        self.eps = eps

        if N is not None:
            self.N = N
            if w is not None:
                self.w = w
            else:
                self.w = np.zeros(self.N)
            if theta is not None:
                self.theta = theta
            else:
                self.theta = 2 * np.pi * np.random.random(size=self.N)
        elif w is not None:
            self.w = w
            self.N = len(w)
            if theta is not None:
                self.theta = theta
            else:
                self.theta = 2 * np.pi * np.random.random(size=self.N)
        elif theta is not None:
            self.theta = theta
            self.N = len(theta)
            self.w = np.zeros(self.N)

    def init_random_walks(self):
        '''
        Genera los paseos al azar
        jump_times es un 1D ndarray con los tiempos en los que se produce un salto (para algún agente)
        random_walks es un 2D ndarray con agente vs posicion (en Z). La cantidad de columnas coincide con len(jump_times)
        '''
        # Generamos los tiempos de salto. Notar que, en promedio, se produce algún salto cada eps/N tiempo
        jump_times = [0]

        jump_times.append(np.random.exponential(self.eps / self.N)) # Primer salto (podría ocurrir más allá del tiempo máximo)
        while jump_times[-1] < self.T:
            jump_times.append(jump_times[-1] + np.random.exponential(self.eps / self.N))

        self.jump_times = np.array(jump_times)

        # Generamos las posiciones de cada agente luego de cada salto
        random_walks = np.zeros((self.N, len(self.jump_times)))
        initial = np.zeros(self.N)
        self.G = nx.Graph()

        for i in range(self.N):
            self.G.add_node(i, pos=np.random.randint(-self.N, self.N)) # Inicializamos los agentes en un punto al azar
            initial[i] = self.G.nodes[i]['pos']

        for combination in KRW.get_combinations(self.N):
            i, j = combination
            if abs(self.G.nodes[i]['pos'] - self.G.nodes[j]['pos']) <= 1:
                self.G.add_edge(i, j)

        self.random_walks = random_walks
        self.random_walks[:,0] = initial

    def derivative(self, theta, t, A):
        '''
        Calcula las derivadas según el modelo:
        dtheta_i
        --------  =   w_i + K * sum_j ( Aij * sin (theta_j - theta_i) ) / N
         dt
        t: compatible con scipy.odeint
        '''
        assert len(theta) == len(self.w), \
            'Las dimensiones no concuerdan'

        theta_i, theta_j = np.meshgrid(theta, theta)
        interactions = sparse.csc_matrix.multiply(A, np.sin(theta_j - theta_i)).sum(axis=0) # Aij * sin(j-i)

        dxdt = self.w + self.K * interactions / self.N # Sumamos sobre las interacciones

        return dxdt

    def integrate(self):
        '''
        Resuelve la ecuación diferencial
        Retorna:
        -------
        historial: 2D ndarray
            Matriz con nodo vs tiempo. Contiene la serie de tiempo de todos los osciladores
        Asume:
        -------
        Se llamada únicamente mediante el método run (requiere inicializar los random walk)
        '''
        historial = np.array([])

        for i in range(len(self.jump_times) - 1):
            timeframe = self.t[np.logical_and(self.jump_times[i] <= self.t, self.t < self.jump_times[i + 1])] # En esta franja de tiempo A es constante

            k = np.random.randint(self.N) # Elegimos al azar al agente que salta
            self.G.nodes[k]['pos'] = self.G.nodes[k]['pos'] + np.random.choice([-1,1], 1, p=[1/2,1/2])

            # Actualizamos los paseos al azar
            new_pos = np.zeros(self.N)
            for n in range(self.N):
                new_pos[n] = self.G.nodes[n]['pos']

            self.random_walks[:,i+1] = new_pos

            # Actualizamos el grafo de manera acorde
            for j in range(self.N):
                if abs(self.G.nodes[k]['pos'] - self.G.nodes[j]['pos']) <= 1 and j != k:
                    self.G.add_edge(k, j)
                elif self.G.has_edge(k, j):
                    self.G.remove_edge(k, j)

            A = nx.adjacency_matrix(self.G)

            if historial.size: # No es la primera iteración
                theta = historial[-1,:] # Tomo de condición inicial el final de la iteración previa
                historial = np.vstack((historial, odeint(self.derivative, theta, timeframe, args=(A,))))
                if KRW.delta(theta) < 0.01: #Tolerancia prefijada
                    break
            else:
                historial = odeint(self.derivative, self.theta, timeframe, args=(A,))

        return historial.T  # Trasponemos por consistencia (historial: nodo vs tiempo)

    def run(self):
        '''
        Retorna:
        -------
        historial: 2D ndarray
            Matriz con nodo vs tiempo. Contiene la serie de tiempo de todos los osciladores
        '''
        self.init_random_walks()

        self.t = np.linspace(0, self.T, int(self.T / self.dt))

        return self.integrate()

    @staticmethod
    def phase_coherence(theta):
        '''
        Calcula el parametro de orden
        '''
        suma = sum([(np.e ** (1j * theta_i)) for theta_i in theta])
        return abs(suma / len(theta))

    @staticmethod
    def angle_dif(dif):
        '''
        Calcula la distancia geodésica, dada la diferencia entre dos ángulos
        '''
        return np.minimum((2 * np.pi) - abs(dif), abs(dif))

    @staticmethod
    def delta(theta):
        '''
        Calcula el delta máximo entre las fases
        '''
        theta = theta % (2 * np.pi)
        deltas = KRW.angle_dif(theta[:, np.newaxis] - theta)
        return np.max(deltas) - np.min(deltas)

    @staticmethod
    def D(theta):
        '''
        Calcula D(theta), como está definida en la tesis
        '''
        theta = theta % (2 * np.pi)
        deltas = KRW.angle_dif(theta[:, np.newaxis] - theta)

        return np.sum(deltas) / 2

    @staticmethod
    def get_combinations(N):
        '''
        Retorna una lista con todos los posibles pares entre N elementos
        '''
        combinations = list()
        
        for i in range(0, N):
            for j in range(i + 1, N):
                combinations.append([i, j])

        return combinations

    def plot_activity(self, historial):
        '''
        Plotea fase vs tiempo por cada oscilador
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t)
        '''
        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.t, historial.T)
        ax.set_xlabel('Tiempo', fontsize=25)
        ax.set_ylabel(r'$\theta$', fontsize=25)

        return ax

    def plot_phase_coherence(self, historial):
        '''
        Plotea el parámetro de orden vs tiempo
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t)
        '''
        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.t, [KRW.phase_coherence(vec) for vec in historial.T], '-', markersize=8)
        ax.set_ylabel('Orden', fontsize=25)
        ax.set_xlabel('Tiempo', fontsize=25)
        ax.set_ylim((-0.05, 1.05))

        return ax

    def plot_D(self, historial):
        '''
        Plotea la función D vs tiempo
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t)
        '''
        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.t, [KRW.D(vec) for vec in historial.T], '-', markersize=8)
        ax.set_ylabel(r'D($\theta$)', fontsize=25)
        ax.set_xlabel('Tiempo', fontsize=25)

        return ax

    def plot_connectivity(self, full_rw):
        '''
        Plotea la conectividad vs tiempo
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t)
        '''
        _, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.t, [KRW.conectivity(vec) for vec in full_rw], '-', markersize=8)
        ax.set_ylabel('Conectividad', fontsize=25)
        ax.set_xlabel('Tiempo', fontsize=25)
        ax.set_ylim((-0.05, 1.05))

        return ax

    def plot_snapshots(self, historial):
        '''
        Plotea tres snapshots del modelo, a tiempos 0, T/2 y T
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t)
        '''
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12, 4),
                        subplot_kw={"ylim": (-1.1, 1.1),
                                    "xlim": (-1.1, 1.1),
                                    "xlabel": r'$\cos(\theta)$',
                                    "ylabel": r'$\sin(\theta)$',})

        times = [0, 1/2, 1]

        for ax, time in zip(axes, times):
            for i in range(self.N):
                ax.plot(np.cos(historial[i, int(time * (len(self.t) - 1))]),
                        np.sin(historial[i, int(time * (len(self.t) - 1))]), 'o', markersize=10)
            ax.set_title(f'Tiempo = {round(self.t[int(time * (len(self.t) - 1))], 2)}')
            background = plt.Circle((0, 0), 1, color='grey', fill=False)
            ax.add_patch(background)

        fig.tight_layout()

        return ax

    def get_full_random_walk(self):
        '''
        Devuelve la posición de agente para cada tiempo
        Retorna:
        -------
            full_rw 2D ndarray
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t y los random walks)
        '''
        full_rw = np.zeros((len(self.t), self.N))
        rw = np.array(self.random_walks)
        aux = 0
        for i in range(len(self.t)):
            while self.t[i] > self.jump_times[aux]:
                aux = aux + 1
            full_rw[i] = self.random_walks[:, aux]

        return full_rw

    def plot_random_walks(self):
        '''
        Plotea el paseo al azar que realiza cada agente
        Retorna:
        -------
            matplotlib axis
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t y los random walks)
        '''
        _, ax = plt.subplots(figsize=(12, 4))
        full_rw = self.get_full_random_walk()

        ax.plot(self.t, full_rw, '-')
        ax.set_xlabel('Tiempo', fontsize=25)

        return ax

    def animate(self, historial):
        '''
        Anima la fase de cada oscilador en función del tiempo
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            La animación
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t y los random walks)
        '''
        fig, ax = plt.subplots(figsize=(4,4))
        ax.cla()
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        ax.set_xlabel(r'$\cos(\theta)$', fontsize=25)
        ax.set_ylabel(r'$\sin(\theta)$', fontsize=25)
        ax.axis('equal')
        background = plt.Circle((0, 0), 1, color='grey', fill=False)
        ax.add_patch(background)

        phases = [plt.plot([], [], 'o', lw=2, ms=8)[0] for _ in range(self.N)]
        txt = ax.text(-.5, 1.2, '', fontsize=15)
        combinations = KRW.get_combinations(self.N)
        lines = [plt.plot([], [], '-.', color='grey', lw=1, ms=8)[0] for _ in combinations]

        full_rw = self.get_full_random_walk()

        def init():
            for phase in phases:
                phase.set_data([], [])
            return phases

        def ani(n):
            txt.set_text(f"Tiempo: {round(self.t[n], 2)}")
            for i, phase in enumerate(phases):
                phase.set_data(np.cos(historial[i,n]), np.sin(historial[i,n]))

            for k, line in enumerate(lines):
                i, j = combinations[k]
                if abs(full_rw[n][i] - full_rw[n][j]) <= 1: # Recordar que esto podría flexibilizarse
                    line.set_data(np.array([[np.cos(historial[i,n]), np.cos(historial[j,n])], [np.sin(historial[i,n]), np.sin(historial[j,n])]]))
                else:
                    line.set_data([], [])

            return phases

        return FuncAnimation(fig, ani, init_func=init, frames=range(0, len(historial[0,:]), int(len(historial[0,:]) / 500)))

    def makeGIF(self, historial):
        '''
        Crea un GIF con la fase de cada oscilador en función del tiempo
        historial: 2D ndarray
            Serie de tiempo, nodo vs tiempo; ie output de KRW.run()
        Retorna:
        -------
            Guarda el GIF en una carpeta de Google Drive
        Asume:
        -------
        Previamente se llamó al método run (requiere inicializar t y los random walks)
        El path para guardar tanto los PNG como el GIF está hardcodeado. Cambiar según corresponda
        '''
        fig, ax = plt.subplots(figsize=(8,8))
        ax.cla()
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        ax.axis('equal')
        background = plt.Circle((0, 0), 1, color='grey', fill=False)
        ax.add_patch(background)

        phases = [plt.plot([], [], 'o', lw=2, ms=8)[0] for _ in range(self.N)]
        for phase in phases:
            phase.set_data([], [])
        txt = ax.text(-.5, 1.2, '', fontsize=15)
        combinations = KRW.get_combinations(self.N)
        lines = [plt.plot([], [], '-.', color='grey', lw=1, ms=8)[0] for _ in combinations]

        full_rw = self.get_full_random_walk()

        def create_frame(n):
            txt.set_text(f"Tiempo: {round(self.t[n], 2)}")

            for i, phase in enumerate(phases):
                phase.set_data(np.cos(historial[i,n]), np.sin(historial[i,n]))

            for k, line in enumerate(lines):
                i, j = combinations[k]
                if abs(full_rw[n][i] - full_rw[n][j]) <= 1: # Recordar que esto podría flexibilizarse
                    line.set_data(np.array([[np.cos(historial[i,n]), np.cos(historial[j,n])], [np.sin(historial[i,n]), np.sin(historial[j,n])]]))
                else:
                    line.set_data([], [])

            plt.savefig(f'/content/gdrive/My Drive/tesis_img/img_{n}.png',
                        transparent = False,
                        facecolor = 'white'
                       )
            #plt.close()

        frames = []

        for n in range(len(self.t)):
            if n % 2**int(np.log10(n+1)+3) == 0:   # No hago todos los frames. Uso escala de tiempo logaritmica (acelero a medida que pasa el tiempo)
                create_frame(n)
                image = imageio.v2.imread(f'/content/gdrive/My Drive/tesis_img/img_{n}.png')
                frames.append(image)

        imageio.mimsave('/content/gdrive/My Drive/tesis_gif/example.gif',
                        frames,
                        fps = 32,
                        loop = 1)

        return True
