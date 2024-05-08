document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.querySelector(".message_input");
    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });
});

(function () {
    // Define a JavaScript function for creating a message
    var Message;
    Message = function (arg) {
        this.text = arg.text, this.message_side = arg.message_side, this.buttons = arg.buttons;
        this.draw = function (_this) {
            return function () {
                var $message;
                $message = $($('.message_template').clone().html());
                $message.addClass(_this.message_side).find('.text').html(_this.text);

                // Add buttons
                var $buttonsContainer = $message.find('.text_wrapper');
                _this.buttons.forEach(function (button) {
                    var $button = $('<button class="message_button">' + button.text + '</button>');
                    $button.click(function () {
                        // Handle button click event here
                        $('.message_input').val(button.text)
                    });
                    $buttonsContainer.append($button);
                });

                // Add copy icon to bot messages
                if (_this.message_side === 'left') {
                    var $copyIcon = $('<img class="copy-icon" src="../static/copy_icon.png" alt="Copy message" title="Copy message">');
                    $copyIcon.css({
                        'width': '20px',   // Adjust the width of the icon
                        'height': '20px',  // Adjust the height of the icon
                        'position': 'absolute',
                        'bottom': '10px',
                        'left': '5px'
                    });
                    $copyIcon.click(function () {
                        var textToCopy = $message.find('.text').text();
                        navigator.clipboard.writeText(textToCopy)
                            .then(function () {
                                console.log('Text copied to clipboard');
                            })
                            .catch(function (err) {
                                console.error('Unable to copy text to clipboard: ', err);
                            });
                    });
                    $copyIcon.css('cursor', 'pointer'); // Set cursor to pointer
                    $copyIcon.hover(function () {
                        $(this).css('cursor', 'pointer'); // Change cursor to pointer on hover
                    }, function () {
                        $(this).css('cursor', 'default'); // Revert cursor to default when not hovered
                    });
                    $message.find('.text').append($copyIcon);
                }

                $('.messages').append($message);
                return setTimeout(function () {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };

    // Execute when the document is ready
    $(function () {

        // Function to get the text of the message
        function getMessageText() {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        }

        // Function to send a message
        function sendMessage(text) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $('.message_input').css('height', 'auto');

            $messages = $('.messages');

            // Set message_side based on whether the message is from the user or chatbot
            var message_side = 'right';

            message = new Message({
                text: text,
                message_side: message_side,
                buttons: [] // No buttons for user message
            });

            // Draw user message
            message.draw();

            // Get the selected option value
            var ChatMode = $('#chatType').val();

            // Call getResponse() to get the chatbot's response
            $.get("/get", {msg: text, option: ChatMode}).done(function (data) {

                var botMessage = new Message({
                    text: marked.parse(data),
                    message_side: 'left',
                    buttons: [] // No buttons for user message

                });

                // Draw bot message
                botMessage.draw();
                $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
            });

            return $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
        }


        $('.fa-arrow-up').click(function (e) {
            return sendMessage(getMessageText());
        });

        // Event listener for pressing enter the message input field
        $('.message_input').keyup(function (e) {
            if (e.which === 13) {
                return sendMessage(getMessageText());
            }
        });

        // Add "Writing..." message
        var writingMessage = new Message({
            //PATENT PROMPT
            text: marked.parse("<p>Hello! I'm your new virtual bot. </p> " +
                "<p>Below are some examples of questions I can address. Don't hesitate to ask your own.</p>" +
                " <p> How can I assist you today?</p>" +

                '\n'
            ),
            //  //LEGAL PROMPT
            // text: marked.parse("<p>Hello! I'm Amanda, your new virtual compliance counsel. As a member of the Ethics " +
            //     "and Compliance team, I'm here to assist with inquiries about our Stora Enso Code, Purpose and Values, " +
            //     "Business Practice Policy, and other internal policies. </p> " +
            //     "<p>Below are some examples of questions I can address. Don't hesitate to ask your own.</p>" +
            //     " <p> How can I assist you today?</p>" +
            //
            //     '\n'
            // ),
            message_side: 'left',

            //PATENT PROMPT
            buttons: [ // Add buttons for initial bot message
                {text: "Tell me  about A METHOD FOR PRODUCING A FILM COMPRISING MICROFIBRILLATED CELLULOSE"}
            ]
            // //LEGAL PROMPT
            // buttons: [ // Add buttons for initial bot message
            //     {text: "Explain what a valid IPR of third parties is"},
            //     {text: "Can you explain Joint Purchasing to me?"},
            //     {text: "Which are the main competitors of Stora Enso in 2024?"},
            //     {text: "How did the demand for paper products change in 2024 compared to the previous year?"},
            // ]
        });
        writingMessage.draw();

    });
}());
