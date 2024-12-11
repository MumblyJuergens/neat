#pragma once

namespace neat
{
    /**
     * A completely pointless class to pass around user data pointers.
     *
     * The *only* thing worth mentioning is Population takes no ownership
     * of this at all, it is the developers jobs to use it wisely and
     * create, destroy, serialize it as needed.
     */
    class UserData
    {
        public:

        virtual ~UserData() = default;
    };

} // End namespace neat.