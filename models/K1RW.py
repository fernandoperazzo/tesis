class K1RW:

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
                self.w = np.zeros(self.N+1)
            if theta is not None:
                self.theta = theta
            else:
                self.theta = 2 * np.pi * np.random.random(size=self.N+1)
        elif w is not None:
            self.w = w
            self.N = len(w)-1
            if theta is not None:
                self.theta = theta
            else:
                self.theta = 2 * np.pi * np.random.random(size=self.N+1)
        elif theta is not None:
            self.theta = theta
            self.N = len(theta)-1
            self.w = np.zeros(self.N+1)

    def init_random_walk(self):
        '''
        Genera los paseos al azar
        jump_times es un 1D ndarray con los tiempos en los que se produce un salto (para algún agente)
        random_walks es un 2D ndarray con agente vs posicion (en Z). La cantidad de columnas coincide con len(jump_times)
        MENTIRA AHORA ES UNA LISTA, MAS ADELANTE SE CONVIERTE EN UN ARRAY
        '''
        # Generamos los tiempos de salto. Notar que, en promedio, se produce un salto cada eps tiempo (porque hay un solo agente moviendose)
        jump_times = [0]

        jump_times.append(np.random.exponential(self.eps)) # Primer salto (podría ocurrir más allá del tiempo máximo)
        while jump_times[-1] < self.T:
            jump_times.append(jump_times[-1] + np.random.exponential(self.eps))

        self.jump_times = np.array(jump_times)

        # Generamos las posiciones de cada agente luego de cada salto

        random_walk = np.zeros(len(self.jump_times))
        self.G = nx.circulant_graph(self.N, [1])

        initial_pos = np.random.randint(self.N)

        random_walk[0] = initial_pos
        self.G.add_node(self.N, pos=initial_pos)
        k = initial_pos % self.N
        self.G.add_edge(k, self.N)

        self.random_walk = random_walk

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

        dxdt = self.w + 1 * interactions / self.N # Sumamos sobre las interacciones; Antes teniamos K como parametro

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

            self.G.nodes[self.N]['pos'] = self.G.nodes[self.N]['pos'] + np.random.choice([-1,1], 1, p=[1/2,1/2])

            # Actualizamos los paseos al azar
            self.random_walk[i+1] = self.G.nodes[self.N]['pos']

            # Actualizamos el grafo de manera acorde
            u, v = list(self.G.edges(self.N))[0]
            self.G.remove_edge(u, v)
            k = self.G.nodes[self.N]['pos'] % self.N
            self.G.add_edge(k[0], self.N,  weight=self.K)

            A = nx.adjacency_matrix(self.G)

            if historial.size: # No es la primera iteración
                theta = historial[-1,:] # Tomo de condición inicial el final de la iteración previa
                historial = np.vstack((historial, odeint(self.derivative, theta, timeframe, args=(A,))))
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
        self.init_random_walk()

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
    def get_combinations(N):
        '''
        Retorna una lista con todos los posibles pares entre N elementos
        '''
        combinations = list()

        for i in range(0, N):
            for j in range(i + 1, N):
                combinations.append([i,j])

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
        ax.plot(self.t, [K1RW.phase_coherence(vec) for vec in historial.T], 'o', markersize=8)
        ax.set_ylabel('Orden', fontsize=25)
        ax.set_xlabel('Tiempo', fontsize=25)
        ax.set_ylim((-0.05, 1.05))

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
        ax.plot(self.t, [K1RW.conectivity(vec) for vec in full_rw], 'o', markersize=8)
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
                        subplot_kw={
                                    "ylim": (-1.1, 1.1),
                                    "xlim": (-1.1, 1.1),
                                    "xlabel": r'$\cos(\theta)$',
                                    "ylabel": r'$\sin(\theta)$',})

        times = [0, 1/2, 1]

        for ax, time in zip(axes, times):
            for i in range(self.N+1):
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
        full_rw = np.zeros(len(self.t))
        rw = np.array(self.random_walk)
        aux = 0
        for i in range(len(self.t)):
            while self.t[i] > self.jump_times[aux]:
                aux = aux + 1
            full_rw[i] = self.random_walk[aux]

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

        phases = [plt.plot([], [], 'o', lw=2, ms=8)[0] for _ in range(self.N+1)]
        txt = ax.text(-.5, 1.2, '', fontsize=15)
        lines = [plt.plot([], [], '-.', color='grey', lw=1, ms=8)[0] for _ in range(self.N+1)]

        full_rw = self.get_full_random_walk()

        def init():
            for phase in phases:
                phase.set_data([], [])
            return phases

        def ani(n):
            txt.set_text(f"Tiempo: {round(self.t[n], 2)}")
            for i, phase in enumerate(phases):
                phase.set_data(np.cos(historial[i,n]), np.sin(historial[i,n]))

            for i, line in enumerate(lines):
                if i < self.N: # Recordar que esto podría flexibilizarse
                    j = (i + 1) % self.N
                    line.set_data(np.array([[np.cos(historial[i,n]), np.cos(historial[j,n])], [np.sin(historial[i,n]), np.sin(historial[j,n])]]))
                else:
                    j = int(full_rw[n] % self.N)
                    line.set_data(np.array([[np.cos(historial[i,n]), np.cos(historial[j,n])], [np.sin(historial[i,n]), np.sin(historial[j,n])]]))

            return phases

        return FuncAnimation(fig, ani, init_func=init, frames=range(0, len(historial[0,:]), int(len(historial[0,:]) / 500))) 
