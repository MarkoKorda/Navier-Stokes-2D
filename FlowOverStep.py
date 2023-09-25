import numpy as np 
import matplotlib.pyplot as plt
import cmasher as cmr 
from tqdm import tqdm
from matplotlib.patches import Rectangle

#constants
N_POINTS_Y = 51
ASPECT_RATIO = 10
KINEMATIC_VISCOSITY = 0.005
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 10000
PLOT_EVERY = 100

STEP_HEIGHT_POINTS = 25
STEP_WIDTH_POINTS = 120

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    cell_length = 1.0 / (N_POINTS_Y - 1)

    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x)
    y = np.linspace(0.0, 1.0, N_POINTS_Y)

    X, Y = np.meshgrid(x, y)

    #INITIAL CONDITIONS
    u_prev = np.ones((N_POINTS_Y + 1, n_points_x))
    u_prev[ :(STEP_HEIGHT_POINTS + 1), :] = 0.0
    
    #top edge of the domain
    u_prev[-1, :] = - u_prev[-2, :]

    #top edge of the step
    u_prev[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = - u_prev[(STEP_HEIGHT_POINTS + 1), 1:STEP_WIDTH_POINTS]

    #right edge of the step
    u_prev[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

    #bottom edge of the domain
    u_prev[0, (STEP_WIDTH_POINTS + 1):-1] = - u_prev[1, (STEP_WIDTH_POINTS + 1):-1]

    #inside of the step
    u_prev[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

    v_prev = np.zeros((N_POINTS_Y, n_points_x + 1))

    p_prev = np.zeros((N_POINTS_Y + 1, n_points_x + 1))


    u_tent = np.zeros_like(u_prev)
    u_next = np.zeros_like(u_prev)

    v_tent = np.zeros_like(v_prev)
    v_next = np.zeros_like(v_prev)

    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(15, 10))

    for iter in tqdm(range(N_TIME_STEPS)):
        diffusion_x = KINEMATIC_VISCOSITY * (
            (
                u_prev[1:-1, 2:  ]
                +
                u_prev[2:  , 1:-1]
                + 
                u_prev[1:-1, 0:-2]
                +
                u_prev[0:-2, 1:-1]
                - 4 * 
                u_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )

        convection_x = (
            (
                u_prev[1:-1, 2:  ]**2
                -
                u_prev[1:-1, 0:-2]**2
            ) / (
                2 * cell_length
            )
            +
            (
                v_prev[1:  , 1:-2]
                +
                v_prev[1:  , 2:-1]
                +
                v_prev[ :-1, 1:-2]
                +
                v_prev[ :-1, 2:-1]
            ) / 4
            *
            (
                u_prev[2:  , 1:-1]
                -
                u_prev[0:-2, 1:-1]
            ) / (
                2 * cell_length
            )
        )

        pressure_gradient_x = (
            (
                p_prev[1:-1, 2:-1]
                -
                p_prev[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        u_tent[1:-1, 1:-1] = (
            u_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_x
                +
                diffusion_x
                -
                convection_x
            )
        )

        #BOUNDARY CONDITIONS
        
        #inflow
        u_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        #outflow
        inflow_mass_rate_tent = np.sum(u_tent[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_tent = np.sum(u_tent[1:-1, -2])
        u_tent[1:-1, -1] = u_tent[1:-1, -2] * inflow_mass_rate_tent / outflow_mass_rate_tent

        #top edge of the step
        u_tent[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = - u_tent[(STEP_HEIGHT_POINTS + 1), 1:STEP_WIDTH_POINTS]

        #right edge of the step
        u_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        #bottom edge of the domain
        u_tent[0, (STEP_WIDTH_POINTS + 1):-1] = - u_tent[1, (STEP_WIDTH_POINTS + 1):-1]

        #top edge of the domain
        u_tent[-1, :] = - u_tent[-2, :]

        #inside of the step
        u_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                v_prev[1:-1, 2:  ]
                +
                v_prev[2:  , 1:-1]
                +
                v_prev[1:-1, 0:-2]
                +
                v_prev[0:-2, 1:-1]
                - 
                4 * v_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )

        convection_y = (
            (
                u_prev[2:-1, 1:  ]
                +
                u_prev[2:-1,  :-1]
                + 
                u_prev[1:-2, 1:  ]
                +
                u_prev[1:-2,  :-1]
            ) / 4
            *
            (
                v_prev[1:-1, 2:  ]
                -
                v_prev[1:-1, 0:-2]
            ) / (
                2 * cell_length
            )
            +
            (
                v_prev[2: , 1:-1]**2
                -
                v_prev[0:-2, 1:-1]**2
            ) / (
                2 * cell_length
            )
        )

        pressure_gradient_y = (
            (
                p_prev[2:-1, 1:-1]
                -
                p_prev[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        v_tent[1:-1, 1:-1] = (
            v_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_y
                + 
                diffusion_y
                -
                convection_y
            )
        )

        #BOUNDARY CONDITIONS

        #inflow
        v_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] = - v_tent[(STEP_HEIGHT_POINTS + 1):-1, 1]

        #outflow
        v_tent[1:-1, -1] = v_tent[1:-1, -2]

        #top edge of the step
        v_tent[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        #right edge of the step
        v_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = - v_tent[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]

        #bottom edge of the domain
        v_tent[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        #top edge of the domain
        v_tent[-1, :] = 0.0

        #inside of the step
        v_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0


        divergence = (
            (
                u_tent[1:-1, 1:  ]
                -
                u_tent[1:-1,  :-1]
            ) / (
                cell_length
            )
            +
            (
                v_tent[1:  , 1:-1]
                -
                v_tent[ :-1, 1:-1]
            ) / (
                cell_length
            )
        )

        p_poisson_rhs = divergence / TIME_STEP_LENGTH

        p_correction_prev = np.zeros_like(p_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_correction_next = np.zeros_like(p_correction_prev)
            p_correction_next[1:-1, 1:-1] = 1/4 * (
                p_correction_prev[1:-1, 2:  ]
                +
                p_correction_prev[2:  , 1:-1]
                +
                p_correction_prev[1:-1,  :-2]
                +
                p_correction_prev[ :-2, 1:-1]
                -
                cell_length**2
                *
                p_poisson_rhs
            )

            #BOUNDARY CONDITIONS
            
            # inflow
            p_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 0] = p_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 1]
            
            # outflow
            p_correction_next[1:-1, -1] = - p_correction_next[1:-1, -2]

            # top edge of the step
            p_correction_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = p_correction_next[(STEP_HEIGHT_POINTS + 1), 1:(STEP_WIDTH_POINTS + 1)]

            # right edge of the step
            p_correction_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = p_correction_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
            
            # bottom edge of the domain
            p_correction_next[0, (STEP_WIDTH_POINTS + 1):-1] = p_correction_next[1, (STEP_WIDTH_POINTS + 1):-1]

            # top edge of the domain
            p_correction_next[-1, :] = p_correction_next[-2, :]

            # inside of the step
            p_correction_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

            p_correction_prev = p_correction_next

        p_next = p_prev + p_correction_next

        p_correction_gradient_x = (
            (
                p_correction_next[1:-1, 2:-1]
                -
                p_correction_next[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        u_next[1:-1, 1:-1] = (
            u_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            p_correction_gradient_x
        )

        p_correction_gradient_y = (
            (
                p_correction_next[2:-1, 1:-1]
                -
                p_correction_next[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        v_next[1:-1, 1:-1] = (
            v_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            p_correction_gradient_y
        )

        # U BOUNDARY CONDITIONS
        
        # inflow
        u_next[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        # outflow
        inflow_mass_rate_next = np.sum(u_next[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_next = np.sum(u_next[1:-1, -2])
        u_next[1:-1, -1] = u_next[1:-1, -2] * inflow_mass_rate_next / outflow_mass_rate_next

        # top edge of the step
        u_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] = - u_next[(STEP_HEIGHT_POINTS + 1), 1:STEP_WIDTH_POINTS]

        # right edge of the step
        u_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # bottom edge of the domain
        u_next[0, (STEP_WIDTH_POINTS + 1):-1] = - u_next[1, (STEP_WIDTH_POINTS + 1):-1]

        # top edge of the domain
        u_next[-1, :] = - u_next[-2, :]

        # inside of the step
        u_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0
        

        # V BOUNDARY CONDITIONS

        # inflow
        v_next[(STEP_HEIGHT_POINTS + 1):-1, 0] = - v_next[(STEP_HEIGHT_POINTS + 1):-1, 1]

        # outflow
        v_next[1:-1, -1] = v_next[1:-1, -2]

        # top edge of the step
        v_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        # right edge of the step
        v_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = - v_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]

        # bottom edge of the domain
        v_next[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        # top edge of the domain
        v_next[-1, :] = 0.0

        # inside of the step
        v_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        
        # time advance
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

        """
        inflow_mass_rate_next = np.sum(u_next[1:-1, 0])
        outflow_mass_rate_next = np.sum(u_next[1:-1, -1])
        print(f"Inflow: {inflow_mass_rate_next}")
        print(f"Outflow: {outflow_mass_rate_next}")
        print()
        """

        #visualisation
        if iter % PLOT_EVERY == 0:
            u_centered = (
                (
                    u_next[1:  , :]
                    +
                    u_next[ :-1, :]
                ) / 2
            )
            v_centered = (
                (
                    v_next[:, 1:  ]
                    +
                    v_next[:,  :-1]
                ) / 2
            )
            p_centered = (
                (
                    p_next[1:  , 1: ]
                    +
                    p_next[ :-1, 1:]
                    +
                    p_next[1:  , :-1]
                    +
                    p_next[ :-1, :-1]
                ) / 4
            )

            u_centered[:(STEP_HEIGHT_POINTS + 1), :(STEP_WIDTH_POINTS + 1)] = 0.0
            v_centered[:(STEP_HEIGHT_POINTS + 1), :(STEP_WIDTH_POINTS + 1)] = 0.0

            step_rectangle = plt.gca().add_patch(Rectangle((0,0), STEP_WIDTH_POINTS * cell_length, STEP_HEIGHT_POINTS * cell_length, facecolor="grey", edgecolor="black"))

            
            plt.contourf(
                X,
                Y,
                np.sqrt(u_centered**2 + v_centered**2),
                levels=20,
                cmap=cmr.ember,
                vmin=-1.0,
                vmax=1.5,
            )

            """
            plt.contourf(
                X, 
                Y, 
                p_centered, 
                levels=50,
            )
            """

            cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

            plt.quiver(
                X[::2, ::25],
                Y[::2, ::25],
                u_centered[::2, ::25],
                v_centered[::2, ::25],
                alpha=0.6,
                scale=30.0,
            )
            
            """
            plt.plot(
                5 * cell_length + u_centered[:, 5],
                Y[:, 5],
                color="black",
                linewidth=3,
            )

            plt.plot(
                40 * cell_length + u_centered[:, 40],
                Y[:, 40],
                color="black",
                linewidth=3,
            )

            plt.plot(
                80 * cell_length + u_centered[:, 80],
                Y[:, 80],
                color="black",
                linewidth=3,
            )

            plt.plot(
                180 * cell_length + u_centered[:, 180],
                Y[:, 180],
                color="black",
                linewidth=3,
            )
            """

            plt.xlabel(r"$x$ [m]", labelpad=-10)
            plt.ylabel(r"$y$ [m]")
            plt.draw()
            plt.pause(0.01)
            plt.clf()


    plt.contourf(
        X,
        Y,
        np.sqrt(u_centered**2 + v_centered**2),
        levels=20,
        cmap=cmr.ember,
        vmin=-1.0,
        vmax=1.5,
    )

    cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

    plt.quiver(
        X[::2, ::25],
        Y[::2, ::25],
        u_centered[::2, ::25],
        v_centered[::2, ::25],
        alpha=0.6,
        scale=30.0,
    )

    plt.plot(
        5 * cell_length + u_centered[(STEP_HEIGHT_POINTS):, 5],
        Y[(STEP_HEIGHT_POINTS):, 5],
        color="black",
        linewidth=3,
    )

    plt.plot(
        80 * cell_length + u_centered[(STEP_HEIGHT_POINTS):, 80],
        Y[(STEP_HEIGHT_POINTS):, 80],
        color="black",
        linewidth=3,
    )

    plt.plot(
        200 * cell_length + u_centered[:, 200],
        Y[:, 200],
        color="black",
        linewidth=3,
    )

    plt.plot(
        400 * cell_length + u_centered[:, 400],
        Y[:, 400],
        color="black",
        linewidth=3,
    )

    step_rectangle = plt.gca().add_patch(Rectangle((0,0), STEP_WIDTH_POINTS * cell_length, STEP_HEIGHT_POINTS * cell_length, facecolor="grey", edgecolor="black"))

    plt.xlabel(r"$x$ [m]", labelpad=-10)
    plt.ylabel(r"$y$ [m]")
    plt.show()

    plt.figure(figsize=(15, 10))

    step_rectangle = plt.gca().add_patch(Rectangle((0,0), STEP_WIDTH_POINTS * cell_length, STEP_HEIGHT_POINTS * cell_length, facecolor="grey", edgecolor="black"))

    plt.contourf(
        X,
        Y,
        np.sqrt(u_centered**2 + v_centered**2),
        levels=20,
        cmap=cmr.ember,
        vmin=-1.0,
        vmax=1.5,
    )

    cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

    plt.streamplot(
        X,
        Y,
        u_centered,
        v_centered,
        color="black",
    )
    plt.xlabel(r"$x$ [m]", labelpad=-10)
    plt.ylabel(r"$y$ [m]")
    plt.show()

    """
    plt.figure(figsize=(15, 10))

    step_rectangle = plt.gca().add_patch(Rectangle((0,0), STEP_WIDTH_POINTS * cell_length, STEP_HEIGHT_POINTS * cell_length, facecolor="grey", edgecolor="black"))

    plt.contourf(
        X, 
        Y, 
        p_centered, 
        levels=100,
    )
    plt.colorbar()

    #plt.quiver(X, Y, u_next, v_next, color="black")
    plt.streamplot(
        X, 
        Y, 
        u_centered, 
        v_centered, 
        color="black"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.title("Pressure")
    plt.show()
    """


if __name__ == "__main__":
    main()
