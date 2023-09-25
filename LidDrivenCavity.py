import numpy as np 
import matplotlib.pyplot as plt
import cmasher as cmr 
from tqdm import tqdm

#constants
N_POINTS = 101
DOMAIN_SIZE = 1.0
N_TIME_STEPS = 5000
TIME_STEP_LENGTH = 0.0005
KINEMATIC_VISCOSITY = 0.01
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0
PLOT_EVERY = 50

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    cell_length = DOMAIN_SIZE / (N_POINTS - 1)

    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x, y)

    u_prev = np.zeros((N_POINTS + 1, N_POINTS))
    v_prev = np.zeros((N_POINTS, N_POINTS + 1))
    p_prev = np.zeros((N_POINTS + 1, N_POINTS + 1))


    u_tent = np.zeros_like(u_prev)
    u_next = np.zeros_like(u_prev)

    v_tent = np.zeros_like(v_prev)
    v_next = np.zeros_like(v_prev)

    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(12, 12))

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

        #boundary conditions
        u_tent[1:-1, 0] = 0.0
        u_tent[1:-1, -1] = 0.0
        u_tent[0, :] = - u_tent[1, :]
        u_tent[-1, :] = 2 * HORIZONTAL_VELOCITY_TOP - u_tent[-2, :]


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

        #boundary conditions
        v_tent[1:-1, 0] = - v_tent[1:-1, 1]
        v_tent[1:-1, -1] = - v_tent[1:-1, -2]
        v_tent[0, :] = 0.0
        v_tent[-1, :] = 0.0

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

            #boundary conditions
            p_correction_next[1:-1, 0] = p_correction_next[1:-1, 1]
            p_correction_next[1:-1, -1] = p_correction_next[1:-1, -2]
            p_correction_next[0, :] = p_correction_next[1, :]
            p_correction_next[-1, :] = p_correction_next[-2, :]

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

        #boundary conditions
        u_next[1:-1, 0] = 0.0
        u_next[1:-1, -1] = 0.0
        u_next[0, :] = - u_next[1, :]
        u_next[-1, :] = 2 * HORIZONTAL_VELOCITY_TOP - u_next[-2, :]

        v_next[1:-1, 0] = - v_next[1:-1, 1]
        v_next[1:-1, -1] = - v_next[1:-1, -2]
        v_next[0, :] = 0.0
        v_next[-1, :] = 0.0

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

            plt.contourf(
                X,
                Y,
                np.sqrt(u_centered**2 + v_centered**2),
                levels=20,
                cmap=cmr.ember,
                vmin=-2.0,
                vmax=5.0,
            )

            """
            plt.contourf(
                X, 
                Y, 
                p_centered, 
                levels=100,
            )
            """

            cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

            plt.quiver(
                X[::4, ::4],
                Y[::4, ::4],
                u_centered[::4, ::4],
                v_centered[::4, ::4],
                alpha=1.0,
            )

            plt.xlabel(r"$x$ [m]", labelpad=-10)
            plt.ylabel(r"$y$ [m]")
            plt.draw()
            plt.pause(0.05)
            plt.clf()

    #plt.figure(figsize=(12, 12))
    
    plt.contourf(
        X,
        Y,
        np.sqrt(u_centered**2 + v_centered**2),
        levels=20,
        cmap=cmr.ember,
        vmin=-2.0,
        vmax=5.0,
    )

    cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")
    
    plt.quiver(
        X[::4, ::4],
        Y[::4, ::4],
        u_centered[::4, ::4],
        v_centered[::4, ::4],
        alpha=1.0,
    )

    plt.xlabel(r"$x$ [m]", labelpad=-10)
    plt.ylabel(r"$y$ [m]")
    plt.show()


    plt.figure(figsize=(12, 12))
    
    plt.contourf(
        X,
        Y,
        np.sqrt(u_centered**2 + v_centered**2),
        levels=20,
        cmap=cmr.ember,
        vmin=-2.0,
        vmax=5.0,
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
    plt.ylim(0.0, 1.0)
    plt.savefig("Cavity0005v.png")
    plt.show()


    plt.figure(figsize=(12, 12))
    
    plt.contourf(
        X,
        Y,
        p_centered,
        levels=20,
    )

    cbar = plt.colorbar(label=r"$p$ [Pa]", orientation="horizontal")

    plt.streamplot(
        X,
        Y,
        u_centered,
        v_centered,
        color="black",
    )

    plt.xlabel(r"$x$ [m]", labelpad=-10)
    plt.ylabel(r"$y$ [m]")
    plt.ylim(0.0, 1.0)
    plt.savefig("Cavity0005p.png")
    plt.show()

if __name__ == "__main__":
    main()
